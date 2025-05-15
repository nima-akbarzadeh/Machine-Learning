import torch
import torch.nn as nn
from torch import optim
from datasets import load_classification_data
from torch.utils.data import DataLoader
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # no activation and no softmax at the end (due to nn.CrossEntropyLoss())
        return out
class Classifier:
    def __init__(self, model_params, hyper_params, dataset_name, device):
        # Data
        train_set, test_set, class_labels = load_classification_data(dataset_name)
        batch_size = hyper_params['batch_size']
        self.train_loader = DataLoader(dataset=train_set,batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
        # Model 
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
        # Other parameters
        self.num_epochs = hyper_params['num_epochs']
        self.clip_grad = hyper_params['clip_grad']
        self.device = device
    def train(self):
        n_total_steps = len(self.train_loader)
        for epoch in range(self.num_epochs):
            losses = []
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(device)
                # Compute the loss
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                losses.append(loss.item())
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
                # Print every 100 optimizer steps
                if (i + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

            mean_loss = sum(losses) / len(losses)
            self.scheduler.step(metrics=mean_loss)

            torch.save(self.model.state_dict(), "models/classifier_FCN.pth")

    def test(self):
        self.model.load_state_dict(torch.load("models/classifier_FCN.pth"))
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
        'clip_grad': True,
    }

    # Dataset
    dataset_name = 'mnist'

    # Training & Testing
    IMG_CLASS = Classifier(model_parameters, hyper_parameters, dataset_name, device)
    IMG_CLASS.train()
    IMG_CLASS.test()