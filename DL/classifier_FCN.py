import torch
import torch.nn as nn
from torch import optim
from datasets import load_classification_data
from torch.utils.data import DataLoader


# ToDo: Missing k-fold cross validation
# ToDo: Missing hyper-parameter tuning


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.initialize_weights()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # no activation and no softmax at the end (due to nn.CrossEntropyLoss())
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
    def __init__(self, model_params, hyper_params, dataset_name, device):

        # Dataset loader
        train_set, test_set, class_labels = load_classification_data(dataset_name)
        batch_size = hyper_params['batch_size']
        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=test_set,
                                      batch_size=batch_size,
                                      shuffle=False)

        # Model initialization
        input_size = 28 * 28
        num_classes = len(class_labels)
        hidden_size = model_params['hidden_size']
        self.model = NeuralNet(input_size, hidden_size, num_classes).to(device)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        learning_rate = hyper_params['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Scaler: helps to reduce RAM and train faster
        self.scaler = torch.cuda.amp.GradScaler()

        # Scheduler
        # When a metric stopped improving for 'patience' number of epochs, the learning rate is reduced by a factor of 2-10.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=0.1*hyper_params['num_epochs'], factor=0.5, verbose=True)
        # # Reduce the learning rate every num_epochs/10 by 0.75
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=hyper_params['num_epochs'], gamma=0.75, verbose=True)

        # Other parameters
        self.num_epochs = hyper_params['num_epochs']
        self.device = device

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
                # Convert it to proper size: [n_batch, 784]
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(device)

                # Forward pass
                if self.device == 'cpu':
                    # Compute the loss
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    losses.append(loss.item())

                    # Backward and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                else:

                    with torch.cuda.amp.autocast():
                        # Compute the loss
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        losses.append(loss.item())

                        # Backward and optimize
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

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
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')


# Running the code
if __name__ == "__main__":

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters
    model_parameters = {
        'hidden_size': 500,
    }

    # Hyper parameters
    hyper_parameters = {
        'num_epochs': 2,
        'batch_size': 100,
        'learning_rate': 0.001,
    }

    # Dataset
    dataset_name = 'mnist'

    # Training & Testing
    IMG_CLASS = Classifier(model_parameters, hyper_parameters, dataset_name, device)
    IMG_CLASS.train()
    IMG_CLASS.test()
