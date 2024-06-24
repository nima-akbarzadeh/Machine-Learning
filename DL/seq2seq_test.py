import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy


# Should get the whole input at once
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers, pr_dropout, device):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.pr_dropout = pr_dropout

        self.embedding = nn.Embedding(input_size, embedding_size, device=device)
        self.dropout = nn.Dropout(p=pr_dropout)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers, dropout=pr_dropout, bidirectional=False, device=device)

    def forward(self, x):

        _, encoder_state = self.lstm(self.dropout(self.embedding(x)))

        return encoder_state


# Should generate single words sequentially
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers, pr_dropout, device):
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.pr_dropout = pr_dropout

        self.embedding = nn.Embedding(input_size, embedding_size, device=device)
        self.dropout = nn.Dropout(p=pr_dropout)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers, dropout=pr_dropout, bidirectional=False, device=device)
        self.fc = nn.Linear(hidden_size, output_size, device=device)

    def forward(self, x, encoder_state):
        x = x.unsqueeze(dim=0)
        output, decoder_state = self.lstm(self.dropout(self.embedding(x)), encoder_state)
        prediction = self.fc(output)

        return prediction.squeeze(dim=0), decoder_state


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source_text, target_text, output_size, force_ratio=0.5):
        encoder_state = self.encoder.forward(source_text)

        target_len = target_text.shape[0]
        batch_size = source_text.shape[1]
        output_size = output_size
        outputs = torch.zeros(target_len, batch_size, output_size).to(device)

        x = target_text[0]
        for t in range(target_len):
            output, decoder_state = self.decoder.forward(x, encoder_state)
            best_prediction = output.argmax(1)
            outputs[t, :, :] = output
            x = target_text[t] if random.random() < force_ratio else best_prediction

        return outputs


class Translator:
    def __init__(self, model_params, hyper_params, tokenizers, device):
        self.test_data = None
        spacy_src = spacy.load(tokenizers[0])
        spacy_tgt = spacy.load(tokenizers[1])

        def tokenizer_ger(text):
            return [tok.text for tok in spacy_src.tokenizer(text)]

        def tokenizer_eng(text):
            return [tok.text for tok in spacy_tgt.tokenizer(text)]

        self.source_language = Field(tokenize=tokenizer_ger, lower=True, init_token="<sos>", eos_token="<eos>")
        self.target_language = Field(tokenize=tokenizer_eng, lower=True, init_token="<sos>", eos_token="<eos>")

        self.train_data, self.valid_data, self.test_data = Multi30k.splits(exts=('.de', '.en'), fields=(self.source_language, self.target_language))

        self.source_language.build_vocab(self.train_data, max_size=10000, min_freq=2)
        self.target_language.build_vocab(self.train_data, max_size=10000, min_freq=2)

        # We're dealing with sentences which have variable lengths
        # Setting sort_within_batch=True, sort_key=lambda x: len(x.src)
        # helps in computation by minimizing the number of paddings in each batch
        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=hyper_params['batch_size'], sort_within_batch=True, sort_key=lambda x: len(x.src), device=device,
        )

        # Model
        input_size_encoder = len(self.source_language.vocab)
        input_size_decoder = len(self.target_language.vocab)
        output_size_decoder = len(self.target_language.vocab)

        encoder = Encoder(
            input_size_encoder, model_params['enc_embed_size'], model_params['hidden_size'], model_params['num_layers'], model_params['enc_dropout'], device
        ).to(device)
        decoder = Decoder(
            input_size_decoder, model_params['dec_embed_size'], model_params['hidden_size'], output_size_decoder, model_params['num_layers'], model_params['dec_dropout'], device
        ).to(device)

        self.model = Seq2Seq(encoder, decoder)

        self.optimizer = optim.Adam(self.model.parameters(), lr=hyper_params['learning_rate'])

        pad_idx = self.target_language.vocab.stoi["<pad>"]
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        # Other parameters
        self.num_epochs = hyper_params['num_epochs']
        self.load_model = hyper_params['load_model']
        self.clip_grad = hyper_params['clip_grad']
        self.device = device


    def train(self):

        for epoch in range(self.num_epochs):

            for batch_idx, batch in enumerate(self.train_iterator):

                print(batch_idx)

                inp_data = batch.src.to(device)

                target = batch.trg.to(device)
                output = self.model(inp_data, target, len(self.target_language.vocab))

                print(target.shape)
                print(output.shape)

                target = target[1:].reshape(-1)
                output = output[1:].reshape(-1, output.shape[2])

                print(target.shape)
                print(output.shape)

                self.optimizer.zero_grad()
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()


    def test(self):

        with torch.no_grad:
            pass

        pass





if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizers = ['de_core_news_sm', 'en_core_web_sm']

    model_params = {
        'hidden_size': 256,  # Needs to be the same for both RNN's
        'num_layers': 2,
        'enc_dropout': 0.5,
        'dec_dropout': 0.5,
        'enc_embed_size': 100,
        'dec_embed_size': 100,
    }

    # Hyper parameters
    hyper_params = {
        'num_epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.001,
        'load_model': False,
        'clip_grad': True,
    }

    GE2EN_Translator = Translator(model_params, hyper_params, tokenizers, device)
    GE2EN_Translator.train()
    GE2EN_Translator.test()
