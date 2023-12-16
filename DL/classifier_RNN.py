import torch
import torch.nn as nn
from torch import optim
from datasets import load_classification_data
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN_ManyToOne(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bi_flag, drop_prob):
        super(RNN_ManyToOne, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bi_flag = bi_flag

        # batch_first=True checks if the batch_size is put as the first dimension
        # In fact, input needs to be: (batch_size, seq_length, input_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True,  bidirectional=self.bi_flag, dropout=drop_prob)

        if bi_flag:
            self.fc = nn.Linear(2 * hidden_size, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

        self.initialize_weights()

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, input_size)
        # x: (n_batch, 28, 28)

        # Set initial hidden states
        if self.bi_flag:
            h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(device)
            # h0: tensor of shape (2*num_layers, batch_size, hidden_size)
            # h0: (4, n_batch, 128)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            # h0: tensor of shape (num_layers, batch_size, hidden_size)
            # h0: (2, n_batch, 128)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n_batch, 28, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n_batch, 128)

        # Fully-connected layer
        out = self.fc(out)
        # out: (n_batch, 10)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.RNN):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GRU):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LSTM):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class RNN_ManyToMany(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length, num_classes, bi_flag, drop_prob):
        super(RNN_ManyToMany, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bi_flag = bi_flag

        # batch_first=True checks if the batch_size is put as the first dimension
        # In fact, input needs to be: (batch_size, seq_length, input_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=self.bi_flag, dropout=drop_prob)

        if bi_flag:
            self.fc = nn.Linear(2 * hidden_size * sequence_length, num_classes)
        else:
            self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

        self.initialize_weights()

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, input_size)
        # x: (n_batch, 28, 28)

        # Set initial hidden states
        if self.bi_flag:
            h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(device)
            # h0: tensor of shape (2*num_layers, batch_size, hidden_size)
            # h0: (4, n_batch, 128)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            # h0: tensor of shape (num_layers, batch_size, hidden_size)
            # h0: (2, n_batch, 128)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n_batch, 28, 128)

        # Flatten the output
        out = out.reshape(out.shape[0], -1)
        # out: (n_batch, 128 * 28)

        # Fully-connected layer
        out = self.fc(out)
        # out: (n_batch, 10)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.RNN):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GRU):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LSTM):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class LSTM_ManyToOne(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bi_flag, drop_prob):
        super(LSTM_ManyToOne, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bi_flag = bi_flag

        # batch_first=True checks if the batch_size is put as the first dimension
        # In fact, input needs to be: (batch_size, seq, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=self.bi_flag, dropout=drop_prob)

        if bi_flag:
            self.fc = nn.Linear(2 * hidden_size, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

        self.initialize_weights()

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, input_size)
        # x: (n_batch, 28, 28)

        # Set initial hidden states (and cell states for LSTM)
        if self.bi_flag:
            h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(device)
            # h0: tensor of shape (2*num_layers, batch_size, hidden_size)
            # c0: tensor of shape (2*num_layers, batch_size, hidden_size)
            # h0: (4, n_batch, 128)
            # h0: (4, n_batch, 128)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            # h0: tensor of shape (num_layers, batch_size, hidden_size)
            # c0: tensor of shape (num_layers, batch_size, hidden_size)
            # h0: (2, n_batch, 128)
            # h0: (2, n_batch, 128)

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n_batch, 28, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n_batch, 128)

        # Fully-connected layer
        out = self.fc(out)
        # out: (n_batch, 10)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.RNN):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GRU):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LSTM):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class LSTM_ManyToMany(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length, num_classes, bi_flag, drop_prob):
        super(LSTM_ManyToMany, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bi_flag = bi_flag

        # batch_first=True checks if the batch_size is put as the first dimension
        # In fact, input needs to be: (batch_size, seq, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=self.bi_flag, dropout=drop_prob)

        if bi_flag:
            self.fc = nn.Linear(2 * hidden_size * sequence_length, num_classes)
        else:
            self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

        self.initialize_weights()

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, input_size)
        # x: (n_batch, 28, 28)

        # Set initial hidden & cell states
        if self.bi_flag:
            h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(device)
            # h0: tensor of shape (2*num_layers, batch_size, hidden_size)
            # c0: tensor of shape (2*num_layers, batch_size, hidden_size)
            # h0: (4, n_batch, 128)
            # h0: (4, n_batch, 128)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            # h0: tensor of shape (num_layers, batch_size, hidden_size)
            # c0: tensor of shape (num_layers, batch_size, hidden_size)
            # h0: (2, n_batch, 128)
            # h0: (2, n_batch, 128)

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n_batch, 28, 128)

        # Flatten the output
        out = out[:, -1, :]
        # out: (n_batch, 128 * 28)

        # Fully-connected layer
        out = self.fc(out)
        # out: (n_batch, 10)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.RNN):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GRU):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LSTM):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class GRU_ManyToOne(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bi_flag, drop_prob):
        super(GRU_ManyToOne, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bi_flag = bi_flag

        # batch_first=True checks if the batch_size is put as the first dimension
        # In fact, input needs to be: (batch_size, seq, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=self.bi_flag, dropout=drop_prob)

        if bi_flag:
            self.fc = nn.Linear(2 * hidden_size, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

        self.initialize_weights()

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, input_size)
        # x: (n_batch, 28, 28)

        # Set initial hidden states (and cell states for LSTM)
        if self.bi_flag:
            h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(device)
            # h0: tensor of shape (2*num_layers, batch_size, hidden_size)
            # h0: (4, n_batch, 128)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            # h0: tensor of shape (num_layers, batch_size, hidden_size)
            # h0: (2, n_batch, 128)

        # Forward propagate RNN
        out, _ = self.gru(x, h0)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n_batch, 28, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n_batch, 128)

        # Fully-connected layer
        out = self.fc(out)
        # out: (n_batch, 10)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.RNN):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GRU):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LSTM):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class GRU_ManyToMany(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length, num_classes, bi_flag, drop_prob):
        super(GRU_ManyToMany, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bi_flag = bi_flag

        # batch_first=True checks if the batch_size is put as the first dimension
        # In fact, input needs to be: (batch_size, seq_length, input_size)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=self.bi_flag, dropout=drop_prob)

        if bi_flag:
            self.fc = nn.Linear(2 * hidden_size * sequence_length, num_classes)
        else:
            self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

        self.initialize_weights()

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, input_size)
        # x: (n_batch, 28, 28)

        # Set initial hidden states
        if self.bi_flag:
            h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(device)
            # h0: tensor of shape (2*num_layers, batch_size, hidden_size)
            # h0: (4, n_batch, 128)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            # h0: tensor of shape (num_layers, batch_size, hidden_size)
            # h0: (2, n_batch, 128)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n_batch, 28, 128)

        # Flatten the output
        out = out.reshape(out.shape[0], -1)
        # out: (n_batch, 128 * 28)

        # Fully-connected layer
        out = self.fc(out)
        # out: (n_batch, 10)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.RNN):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GRU):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LSTM):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


# Model training and prediction
class Classifier:
    def __init__(self, parameters, dataset_name):

        # Dataset loader
        train_set, test_set, class_labels = load_classification_data(dataset_name)
        batch_size = parameters['batch_size']
        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=test_set,
                                      batch_size=batch_size,
                                      shuffle=False)

        # Model initialization
        # 'self.sequence_length' (28) sequences where each has 'self.input_size' (28) features
        self.input_size = 28
        self.sequence_length = 28
        num_classes = len(class_labels)
        hidden_size = parameters['hidden_size']
        self.model = RNN_ManyToOne(self.input_size, hidden_size, parameters['num_layers'],
                                   num_classes, parameters['bidirect_flag'], parameters['dropout_prob']).to(device)
        # self.model = RNN_ManyToMany(self.input_size, hidden_size, parameters['num_layers'], self.sequence_length,
        #                             num_classes, parameters['bidirect_flag'], parameters['dropout_prob']).to(device)
        # self.model = LSTM_ManyToOne(self.input_size, hidden_size, parameters['num_layers'],
        #                             num_classes, parameters['bidirect_flag'], parameters['dropout_prob']).to(device)
        # self.model = LSTM_ManyToMany(self.input_size, hidden_size, parameters['num_layers'], self.sequence_length,
        #                              num_classes, parameters['bidirect_flag'], parameters['dropout_prob']).to(device)
        # self.model = GRU_ManyToOne(self.input_size, hidden_size, parameters['num_layers'],
        #                            num_classes, parameters['bidirect_flag'], parameters['dropout_prob']).to(device)
        # self.model = GRU_ManyToMany(self.input_size, hidden_size, parameters['num_layers'], self.sequence_length,
        #                             num_classes, parameters['bidirect_flag'], parameters['dropout_prob']).to(device)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        learning_rate = parameters['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Scheduler
        # When a metric stopped improving for 'patience' number of epochs, the learning rate is reduced by a factor of 2-10.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=parameters['lr_update_epochs'], factor=0.5, verbose=True)
        # # Reduce the learning rate every num_epochs/10 by 0.75
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.parameters['num_epochs'], gamma=0.75, verbose=True)

        # Other parameters
        self.num_epochs = parameters['num_epochs']


    def train(self):
        n_total_steps = len(self.train_loader)
        for epoch in range(self.num_epochs):
            # # If loading from a checkpoint is being carried out
            # # 1. Create the model and optimizer
            # # 2. Load the state_dict for each
            # # Since they are part of ImageClassifier, step 1 is skipped
            # checkpoint = torch.load("./checkpoints/classifier_CNN.pth")
            # self.model.load_state_dict(checkpoint["model_state"])
            # self.optimizer.load_state_dict(checkpoint["model_state"])
            losses = []
            for i, (images, labels) in enumerate(self.train_loader):
                # origin shape: [N, 1, 28, 28]
                # resized: [N, 28, 28]
                images = images.reshape(-1, self.sequence_length, self.input_size).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                losses.append(loss.item())

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Print every 100 optimizer steps
                if (i + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

            mean_loss = sum(losses) / len(losses)
            self.scheduler.step(metrics=mean_loss)

            # Two ways to save the models
            torch.save(self.model.state_dict(), "models/classifier_FCN.pth")
            # torch.save(self.models, "./models/classifier_FCN.pth")


    def test(self):
        # # If a saved models is being tested either use
        # # 1. Create the model and load the state_dict
        # # Since the model is part of ImageClassifier, we just load the state_dict
        self.model.load_state_dict(torch.load("models/classifier_CNN.pth"))
        # # 2. Load the whole model
        # self.models = torch.load("./models/classifier_CNN.pth")
        self.model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in self.test_loader:
                images = images.reshape(-1, self.sequence_length, self.input_size).to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')


# Running the code
if __name__ == "__main__":
    # Hyper-parameters
    parameters = {
        'hidden_size': 128,
        'num_layers': 2,
        'num_epochs': 2,
        'batch_size': 100,
        'learning_rate': 0.001,
        'bidirect_flag': False,
        'dropout_prob': 0.25,
    }

    dataset_name = 'mnist'

    IMG_CLASS = Classifier(parameters, dataset_name)
    IMG_CLASS.train()
    IMG_CLASS.test()
