import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from utils import bleu, save_checkpoint, load_checkpoint


# Should get the whole input at once
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, drop_prob):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        print('***************************')
        print(f'input_size: {input_size}')
        print(f'embedding_size: {embedding_size}')
        print(f'hidden_size: {hidden_size}')
        print(f'num_layers: {num_layers}')

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(drop_prob)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=drop_prob)


    def forward(self, x):
        # x: (seq_length, N)
        embedding = self.dropout(self.embedding(x))
        # embedding(x): (seq_length, N, embedding_size)

        _, (hidden, cell) = self.lstm(embedding)

        return hidden, cell


# Should generate single words sequentially
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, drop_prob):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        print('***************************')
        print(f'output_size: {output_size}')

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(drop_prob)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x: (N), Therefore, add one dimension to the input x for single word prediction at a time
        x = x.unsqueeze(dim=0)
        # x: (1, N)

        embedding = self.dropout(self.embedding(x))
        # embedding(x): (1, N, embedding_size)

        output, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        # output: (1, N, hidden_size)

        print(f'decoder lstm output shape {output.shape}')

        prediction = self.fc(output)
        # prediction: (1, N, eng_vocabs_length)

        print(f'decoder fc output shape {prediction.shape}')

        return prediction.squeeze(0), hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, source, target, target_language, force_ratio=0.5):
        target_len = target.shape[0]
        batch_size = source.shape[1]
        target_vocab_size = len(target_language.vocab)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        print(f"=======================================")
        print(f"source shape: {source.shape}")
        print(f"target shape: {target.shape}")
        print(f"target_vocab_size: {target_vocab_size}")

        hidden, cell = self.encoder(source)

        print(f"encoder hidden shape: {hidden.shape}")
        print(f"encoder cell shape: {cell.shape}")

        # Get the start token
        x = target[0]

        print(f"start token shape: {x.shape}")
        print(f"desired outputs shape: {outputs.shape}")

        for t in range(target_len):

            output, hidden, cell = self.decoder(x, hidden, cell)
            # output: (N, eng_vocabs_length)

            print(f'----{t}')
            print(f"decoder output shape: {output.shape}")
            print(f"decoder hidden shape: {hidden.shape}")
            print(f"decoder cell shape: {cell.shape}")

            best_pred = output.argmax(1)
            outputs[t, :, :] = output

            # The LSTM decoder predicts words one by one, and it does it sequentially based on the previous predicted word.
            # This dependency may not be good. Hence, to incease the randomness in the training phase we sometimes let the
            # prediction and sometimes let the target word be the next input to the LSTM.
            x = target[t] if random.random() < force_ratio else best_pred

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
        output_size = len(self.target_language.vocab)
        encoder = Encoder(
            input_size_encoder, model_params['enc_embed_size'],
            model_params['hidden_size'], model_params['num_layers'], model_params['enc_dropout']
        ).to(device)
        decoder = Decoder(
            input_size_decoder, model_params['dec_embed_size'],
            model_params['hidden_size'], output_size, model_params['num_layers'], model_params['dec_dropout'],
        ).to(device)
        self.model = Seq2Seq(encoder, decoder).to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=hyper_params['learning_rate'])

        # Scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # Scheduler
        # When a metric stopped improving for 'patience' number of epochs, the learning rate is reduced by a factor of 2-10.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=0.1*hyper_params['num_epochs'], factor=0.5, verbose=True)
        # # Reduce the learning rate every num_epochs/10 by 0.75
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=hyper_params['num_epochs'], gamma=0.75, verbose=True)

        # As padding is applied to the sentences, we don't want to incur a loss for that
        pad_idx = self.target_language.vocab.stoi["<pad>"]
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

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

                    self.optimizer.zero_grad()
                    loss = self.criterion(output, target)
                    losses.append(loss.item())

                    # Back prop
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


if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizers: The sequence should be [source language, target language]
    tokenizers = ['de_core_news_sm', 'en_core_web_sm']

    # Model parameters
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
