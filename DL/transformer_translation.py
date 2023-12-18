from websockets import exceptions
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator


class Transformer(nn.Module):
    def __init__(
            self, embedding_size, source_cardinality, target_cardinality, source_pad_idx, num_heads,
            num_encoder_layers, num_decoder_layers, expansion, dropout, max_len, device,
    ):
        super(Transformer, self).__init__()
        self.source_word_embedding = nn.Embedding(source_cardinality, embedding_size)
        self.source_position_embedding = nn.Embedding(max_len, embedding_size)
        self.target_word_embedding = nn.Embedding(target_cardinality, embedding_size)
        self.target_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size, num_heads, num_encoder_layers,
            num_decoder_layers, expansion, dropout,
        )
        self.fc_out = nn.Linear(embedding_size, target_cardinality)
        self.dropout = nn.Dropout(dropout)
        self.source_pad_idx = source_pad_idx

    def make_source_mask(self, source):
        # source: (source_len, num_data)
        source_mask = source.transpose(0, 1) == self.source_pad_idx
        # source_mask: (num_data, source_len)

        return source_mask.to(self.device)

    def forward(self, source, target):
        source_seq_length, N = source.shape
        target_seq_length, N = target.shape

        source_positions = (
            torch.arange(0, source_seq_length).unsqueeze(1).expand(source_seq_length, N).to(self.device)
        )

        target_positions = (
            torch.arange(0, target_seq_length).unsqueeze(1).expand(target_seq_length, N).to(self.device)
        )

        embed_source = self.dropout(
            (self.source_word_embedding(source) + self.source_position_embedding(source_positions))
        )
        embed_target = self.dropout(
            (self.target_word_embedding(target) + self.target_position_embedding(target_positions))
        )

        source_padding_mask = self.make_source_mask(source)
        target_mask = self.transformer.generate_square_subsequent_mask(target_seq_length).to(self.device)

        out = self.transformer(
            embed_source, embed_target, source_key_padding_mask=source_padding_mask, tgt_mask=target_mask,
        )
        out = self.fc_out(out)
        return out


class Translator:
    def __init__(self, model_parameters, hyper_params, tokenizers, device):
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
        source_cardinality = len(self.source_language.vocab)
        target_cardinality = len(self.target_language.vocab)
        source_pad_idx = self.source_language.vocab.stoi["<pad>"]
        self.model = Transformer(
            model_parameters['embed_size'], source_cardinality, target_cardinality, source_pad_idx,
            model_parameters['num_heads'], model_parameters['enc_layers'], model_parameters['dec_layers'],
            model_parameters['expansion'], model_parameters['dropout'], model_parameters['max_length'], device,
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=hyper_params['learning_rate'])

        # Scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # Scheduler
        # When a metric stopped improving for 'patience' number of epochs, the learning rate is reduced by a factor of 2-10.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=0.1*hyper_params['num_epochs'], factor=0.5, verbose=True)
        # # Reduce the learning rate every num_epochs/10 by 0.75
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.parameters['num_epochs'], gamma=0.75, verbose=True)

        # As padding is applied to the sentences, we don't want to incur a loss for that
        target_pad_idx = self.target_language.vocab.stoi["<pad>"]
        self.criterion = nn.CrossEntropyLoss(ignore_index=target_pad_idx)

        # Other parameters
        self.num_epochs = hyper_params['num_epochs']
        self.load_model = hyper_params['load_model']
        self.clip_grad = hyper_params['clip_grad']
        self.device = device

        # Tensorboard to get nice loss plot
        self.writer = SummaryWriter(f"runs/loss_plot")


    def train(self):

        if self.load_model:
            load_checkpoint(torch.load("checkpoints/my_checkpoint.pth.tar"), self.model, self.optimizer)

        step = 0
        for epoch in range(self.num_epochs):
            print(f"[Epoch {epoch} / {self.num_epochs}]")

            checkpoint = {"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
            save_checkpoint(checkpoint)

            losses = []
            for batch_idx, batch in enumerate(self.train_iterator):
                # Get input and targets and get to cuda
                inp_data = batch.src.to(device)
                target = batch.trg.to(device)

                # Forward prop
                if str(self.device) == "cpu":
                    # Compute the output
                    output = self.model(inp_data, target, self.target_language)

                    # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
                    # doesn't take input in that form. For example if we have MNIST we want to have
                    # output to be: (N, 10) and targets just (N). Here we can view it in a similar
                    # way that we have output_words * batch_size that we want to send in into
                    # our cost function, so we need to do some reshapin.
                    # [1:] is to remove the start token
                    output = output[1:].reshape(-1, output.shape[2])
                    target = target[1:].reshape(-1)

                    # Compute the loss
                    loss = self.criterion(output, target)
                    losses.append(loss.item())

                    # Back prop
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.clip_grad:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                    self.optimizer.step()

                else:

                    with torch.cuda.amp.autocast():
                        # Compute the loss
                        output = self.model(inp_data, target, self.target_language)
                        output = output[1:].reshape(-1, output.shape[2])
                        target = target[1:].reshape(-1)
                        loss = self.criterion(output, target)
                        losses.append(loss.item())

                        # Backward and optimize
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        if self.clip_grad:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                # Plot to tensorboard
                self.writer.add_scalar("Training loss", loss, global_step=step)
                step += 1

            mean_loss = sum(losses) / len(losses)
            self.scheduler.step(metrics=mean_loss)


    def test(self):
        score = bleu(self.test_data[1:100], self.model, self.source_language, self.target_language, device)
        print(f"Bleu score {score * 100:.2f}")


if __name__ == "__main__":

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # Load tokenizers: The sequence should be [source language, target language]
    tokenizers = ['de_core_news_sm', 'en_core_web_sm']

    model_parameters = {
        'embed_size': 512,
        'num_heads': 8,
        'enc_layers': 3,
        'dec_layers': 3,
        'dropout': 0.1,
        'max_length': 100,
        'expansion': 4,
    }

    hyper_params = {
        'num_epochs': 10000,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'load_model': True,
        'clip_grad': True,
    }

    Transformer_Translator = Translator(model_parameters, hyper_params, tokenizers, device)
    Transformer_Translator.train()
    Transformer_Translator.test()


