import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(SelfAttention, self).__init__()

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        assert (isinstance(self.head_dim, int)), \
            "embedding_size should be divisable by num_heads in SelfAttention block!"

        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc = nn.Linear(self.num_heads * self.head_dim, embedding_size)

        nn.init.kaiming_uniform_(self.keys.weight)
        nn.init.kaiming_uniform_(self.values.weight)
        nn.init.kaiming_uniform_(self.queries.weight)
        nn.init.kaiming_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, keys, values, queries, mask):

        # Get the parameters
        num_data = queries.shape[0]
        keys_len, values_len, queries_len = keys.shape[1], values.shape[1], queries.shape[1]

        # Split the embedding into mutiple heads
        # (num_data, data_len, embedding_size) -> (num_data, data_len, num_heads, head_dim)
        keys = keys.reshape(num_data, keys_len, self.num_heads, self.head_dim)
        values = values.reshape(num_data, values_len, self.num_heads, self.head_dim)
        queries = queries.reshape(num_data, queries_len, self.num_heads, self.head_dim)

        keys = self.keys(keys)
        values = self.values(values)
        queries = self.queries(queries)

        # (eneregies) specify for each element in (queries), how much weight or attention is needed
        # for each element in the (keys), given each (data) in a (head) of the attention block.
        # Therefore, the energies shape should be (num_data, num_heads, queries_len, keys_len)
        energies = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energies = energies.masked_fill(mask == 0, float("-1e20"))

        # Softmax is applied over the keys
        normalized_energies = torch.softmax(energies / (self.embedding_size ** 0.5), dim=3)

        # The attention shape should be (num_data, queries_len, embedding_size)
        # We first create the (num_data, queries_len, num_heads, head_dim) and then reshape it.
        # Note the (keys_len) and the (values_len) should match.
        attention = torch.einsum("nhql,nlhd->nqhd", [normalized_energies, values]).reshape(
            num_data, queries_len, self.num_heads * self.head_dim
        )

        return self.fc(attention)


class TransformerSubBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout, expansion):
        super(TransformerSubBlock, self).__init__()
        self.attention = SelfAttention(embedding_size, num_heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, expansion * embedding_size),
            nn.ReLU(),
            nn.Linear(expansion * embedding_size, embedding_size),
        )
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, keys, values, queries, mask):

        attention = self.attention(keys, values, queries, mask)
        x = self.dropout(self.norm1(attention + queries))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class Encoder(nn.Module):
    def __init__(self, source_cardinality, embedding_size, num_layers, num_heads, expansion, dropout, max_length, device):
        super(Encoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(source_cardinality, embedding_size)
        self.position_embedding = nn.Embedding(max_length, embedding_size)
        self.layers = nn.ModuleList(
            [
                TransformerSubBlock(embedding_size, num_heads, dropout, expansion) for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x, mask):
        # Get the parameters
        num_data, seq_length = x.shape

        # Apply the positional and word embeddings
        positions = torch.arange(0, seq_length).expand(num_data, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        # Create the layers
        for layer in self.layers:
            # The keys, values and queries are all the same in Encoder
            out = layer(out, out, out, mask)

        return out


class DecoderSubBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout, expansion):
        super(DecoderSubBlock, self).__init__()
        self.attention = SelfAttention(embedding_size, num_heads)
        self.norm = nn.LayerNorm(embedding_size)
        self.trans_sub_block = TransformerSubBlock(embedding_size, num_heads, dropout, expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, keys, values, source_mask, target_mask):

        attention = self.attention(x, x, x, target_mask)
        queries = self.dropout(self.norm(attention + x))
        out = self.trans_sub_block(keys, values, queries, source_mask)

        return out


class Decoder(nn.Module):
    def __init__(self, target_cardinality, embedding_size, num_layers, num_heads, expansion, dropout, max_length, device):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_cardinality, embedding_size)
        self.position_embedding = nn.Embedding(max_length, embedding_size)

        self.layers = nn.ModuleList(
            [
                DecoderSubBlock(embedding_size, num_heads, dropout, expansion) for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(embedding_size, target_cardinality)

    def forward(self, x, encoder_output, source_mask, target_mask):
        # Get the parameters
        num_data, seq_length = x.shape

        # Apply the positional and word embeddings
        positions = torch.arange(0, seq_length).expand(num_data, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        # Create the layers
        for layer in self.layers:
            # The keys, values and queries are all the same in Encoder
            x = layer(x, encoder_output, encoder_output, source_mask, target_mask)

        return self.fc(x)


class Transformer(nn.Module):
    def __init__(
        self, source_cardinality, target_cardinality, source_pad_idx, target_pad_idx, embedding_size=512,
            num_layers=6, expansion=4, num_heads=8, dropout=0, max_length=100, device=torch.device("cpu")
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            source_cardinality, embedding_size, num_layers, num_heads, expansion, dropout, max_length, device
        )
        self.decoder = Decoder(
            target_cardinality, embedding_size, num_layers, num_heads, expansion, dropout, max_length, device
        )
        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device

    def make_source_mask(self, source):

        # Source mask shape should be (num_data, 1, 1, source_len)
        source_mask = (source != self.source_pad_idx).unsqueeze(1).unsqueeze(2)

        return source_mask.to(self.device)

    def make_target_mask(self, target):

        num_data, target_len = target.shape
        # Target mask shape should be (num_data, 1, target_len, target_len)
        # The target mask is a lower triangular matrix to mask the future targets
        # in the sequence.
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(
            num_data, 1, target_len, target_len
        )

        return target_mask.to(self.device)

    def forward(self, source, target):

        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)
        encoder_out = self.encoder(source, source_mask)
        out = self.decoder(target, encoder_out, source_mask, target_mask)

        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_cardinality = 10
    target_cardinality = 10
    source_data = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    target_data = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    source_pad_idx = 0
    target_pad_idx = 0
    model = Transformer(source_cardinality, target_cardinality, source_pad_idx, target_pad_idx, device=device).to(device)
    out = model(source_data, target_data[:, :-1])
    print(out.shape)














