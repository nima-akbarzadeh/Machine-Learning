import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from datasets import load_classification_data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ToDo: Missing k-fold cross validation
# ToDo: Missing hyper-parameter tuning
# ToDo: Give warning if (input_size - kernel_size + 2 * conv_padding) / conv_stride is not integer.
# ToDo: Use confusion matrix at the end, and all other scores than accuracy


# Fully connected neural network with one hidden layer
class ConvNet(nn.Module):
    def __init__(self, inp_channels=3, mid_channels=6, out_channels=12, kernel_size=5,
                 conv_padding=0, conv_stride=1, filter_size=2, stride=2, input_size=32,
                 hidden_size=128, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inp_channels,
                               out_channels=mid_channels,
                               kernel_size=tuple([kernel_size, kernel_size]),
                               stride=tuple([conv_stride, conv_stride]))
        input_width = int(1 + (input_size - kernel_size + 2 * conv_padding) / conv_stride)
        self.pool1 = nn.MaxPool2d(kernel_size=filter_size,
                                  stride=stride)
        input_width = int(1 + (input_width - filter_size) / stride)
        self.conv2 = nn.Conv2d(in_channels=mid_channels,
                               out_channels=out_channels,
                               kernel_size=tuple([kernel_size, kernel_size]),
                               stride=tuple([conv_stride, conv_stride]))
        input_width = int(1 + (input_width - kernel_size + 2 * conv_padding) / conv_stride)
        self.pool2 = nn.MaxPool2d(kernel_size=filter_size,
                                  stride=stride)
        input_width = int(1 + (input_width - filter_size) / stride)
        self.fc_input_size = out_channels*input_width*input_width
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

        self.initialize_weights()

    def forward(self, x):  # -> input_size: (n, mid_channels:3, input_width:32, input_width:32)
        x = self.pool1(F.relu(self.conv1(x)))  # -> output_size: (n, mid_channels:6, input_width:14, input_width:14)
        x = self.pool2(F.relu(self.conv2(x)))  # -> output_size: (n, out_channels:16, input_width:5, input_width:5)
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No need for softmax due to nn.CrossEntropyLoss()
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


# Model training and prediction
class ImageClassifier:
    def __init__(self, model_params, hyper_params, dataset_name, writer, device):

        # Dataset loader
        train_set, test_set, self.class_labels = load_classification_data(dataset_name)
        self.batch_size = hyper_params['batch_size']
        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=self.batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=test_set,
                                      batch_size=self.batch_size,
                                      shuffle=False)

        # Model initialization
        self.input_size = 32
        self.num_classes = len(self.class_labels)
        self.hidden_size = model_params['hidden_size']
        self.model = ConvNet(input_size=self.input_size, hidden_size=self.hidden_size, num_classes=self.num_classes).to(device)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        learning_rate = hyper_params['learning_rate']
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        # Scaler: helps to reduce RAM and train faster
        self.scaler = torch.cuda.amp.GradScaler()

        # Scheduler
        # When a metric stopped improving for 'patience' number of epochs, the learning rate is reduced by a factor of 2-10.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=0.1*hyper_params['num_epochs'], factor=0.5, verbose=True)
        # # Reduce the learning rate every num_epochs/10 by 0.75
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=hyper_params['num_epochs'], gamma=0.75, verbose=True)

        # Other parameters
        self.num_epochs = hyper_params['num_epochs']
        self.clip_grad = hyper_params['clip_grad']
        self.device = device

        # Tensorboard
        self.writer = writer
        # self.writer.add_graph(self.model)

    def train(self):
        training_loss = 0.0
        training_correct_preds = 0
        n_total_steps = len(self.train_loader)
        for epoch in tqdm(range(self.num_epochs)):
            # # If loading from a checkpoint is being carried out
            # # 1. Create the model and optimizer
            # # 2. Load the state_dict for each
            # # Since they are part of ImageClassifier, step 1 is skipped
            # checkpoint = torch.load("./checkpoints/classifier_CNN.pth")
            # self.model.load_state_dict(checkpoint["model_state"])
            # self.optimizer.load_state_dict(checkpoint["model_state"])
            losses = []
            for i, (images, labels) in enumerate(self.train_loader):
                # Call the data
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                if self.device == 'cpu':
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

                else:

                    with torch.cuda.amp.autocast():
                        # Compute the loss
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        losses.append(loss.item())

                        # Backward and optimize
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        if self.clip_grad:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                training_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                training_correct_preds += (predicted == labels).sum().item()

                # Print every 2000 optimizer steps
                if (i + 1) % 2000 == 0:
                    print(f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
                    self.writer.add_scalar('training_loss', training_loss / 2000, epoch * n_total_steps + i)
                    self.writer.add_scalar('training_accu', 100 * training_correct_preds / 2000, epoch * n_total_steps + i)
                    training_loss = 0.0
                    training_correct_preds = 0

            mean_loss = sum(losses) / len(losses)
            self.scheduler.step(metrics=mean_loss)

            # Save data if training is interrupted
            checkpoint = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
            }
            torch.save(checkpoint, "./checkpoints/classifier_CNN.pth")

        # Two ways to save the models
        torch.save(self.model.state_dict(), "models/classifier_CNN.pth")
        # torch.save(self.models, "./models/classifier_CNN.pth")

    def test(self):
        # # If a saved models is being tested either use
        # # 1. Create the model and load the state_dict
        # # Since the model is part of ImageClassifier, we just load the state_dict
        self.model.load_state_dict(torch.load("models/classifier_CNN.pth"))
        # # 2. Load the whole model
        # self.models = torch.load("./models/classifier_CNN.pth")
        self.model.eval()

        probs = []
        preds = []
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0] * self.num_classes
            n_class_samples = [0] * self.num_classes
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for i in range(self.num_classes):
                    label = labels[i]
                    pred = predicted[i]
                    if label == pred:
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

                # For computation of precision & recall in Tensorboard
                preds.append(predicted)
                probs.append([F.softmax(output, dim=0) for output in outputs])

            # Computing the overall accuracy
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network: {acc} %')

            # For computation of precision & recall in Tensorboard
            preds = torch.cat(preds)
            probs = torch.cat([torch.stack(batch) for batch in probs])

            for i in range(self.num_classes):
                # Computing the class-specific accuracy
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {self.class_labels[i]}: {acc} %')

                # Adding the precision and recall computation to Tensorboard
                preds_i = preds == i
                probs_i = probs[:, i]
                self.writer.add_pr_curve(str(i), preds_i, probs_i, global_step=0)
                self.writer.close()


# Running the code
if __name__ == "__main__":

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters
    model_params = {
        'hidden_size': 500,
    }

    # Hyper parameters
    hyper_params = {
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'clip_grad': True,
    }

    # Dataset
    dataset_name = 'cifar10'

    # Tensorboard
    writer = SummaryWriter(f"runs/{dataset_name}")

    # # Training & Testing
    IMG_CLASS = ImageClassifier(model_params, hyper_params, dataset_name, writer, device)
    IMG_CLASS.train()
    IMG_CLASS.test()
