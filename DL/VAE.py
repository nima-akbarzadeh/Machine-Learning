import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from datasets import load_classification_data
from torch.utils.data import DataLoader
from torchvision.utils import save_image


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super(VariationalAutoEncoder, self).__init__()

        # We assume the latent space is drawn from Gaussian

        # Encoder
        self.input2hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden2avg = nn.Linear(hidden_dim, output_dim)
        self.hidden2std = nn.Linear(hidden_dim, output_dim)

        # Decoder
        self.latent2hidden = nn.Linear(output_dim, hidden_dim)
        self.hidden2input = nn.Linear(hidden_dim, input_dim)

        # Weight initialization
        self.initialize_weights()

    def encode(self, x):
        # Contruct q_phi(z|x)
        x = F.relu(self.input2hidden(x))
        mu, sigma = self.hidden2avg(x), self.hidden2std(x)
        return mu, sigma

    def decode(self, z):
        # Construct the input from latent variable
        z = F.relu(self.latent2hidden(z))
        return torch.sigmoid(self.hidden2input(z))

    def forward(self, x):
        # The first part
        mu, sigma = self.encode(x)
        # Reconstructing the latent variable from
        # sufficient statistics to backpropagate
        noise = torch.randn_like(sigma)
        z_reparam = mu + sigma * noise
        # The reconstructed input
        x_reconst = self.decode(z_reparam)
        return mu, sigma, x_reconst

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


class Reconstructor:
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
        self.input_size = 28 * 28
        num_classes = len(class_labels)
        hidden_size = model_params['hidden_size']
        self.model = VariationalAutoEncoder(self.input_size, hidden_size, num_classes).to(device)

        # Loss - The output will be summed by setting reduction="sum"
        # BCE can be used for any values in (0-1), e.g., pixels in (0-1)
        self.criterion = nn.BCELoss(reduction="sum")

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
        self.clip_grad = hyper_params['clip_grad']
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
            for i, (images, _) in enumerate(self.train_loader):
                # Convert it to proper size: [n_batch, 784]
                images = images.reshape(-1, 28 * 28).to(self.device)
                # images = images.view(images.shape[0], self.input_size).to(self.device)

                # Forward pass
                if str(self.device) == "cpu":
                    # Compute the loss
                    mu, sigma, outputs = self.model(images)
                    loss_regularization = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
                    loss_reconstruction = self.criterion(outputs, images)
                    loss = loss_regularization + loss_reconstruction
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
                        mu, sigma, outputs = self.model(images)
                        loss_regularization = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
                        loss_reconstruction = self.criterion(outputs, images)
                        loss = loss_regularization + loss_reconstruction
                        losses.append(loss.item())

                        # Backward and optimize
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        if self.clip_grad:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                # Print every 100 optimizer steps
                if (i + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

            mean_loss = sum(losses) / len(losses)
            self.scheduler.step(metrics=mean_loss)

            # Two ways to save the models
            torch.save(self.model.state_dict(), "models/reconstructor_VAE.pth")
            # torch.save(self.models, "./models/reconstructor_VAE.pth")

    def test(self):
        # # If a saved models is being tested either use
        # # 1. Create the model and load the state_dict
        # # Since the model is part of ImageClassifier, we just load the state_dict
        self.model.load_state_dict(torch.load("models/reconstructor_VAE.pth"))
        # # 2. Load the whole model
        # self.models = torch.load("./models/reconstructor_VAE.pth")
        self.model.eval()
        with torch.no_grad():
            for images, _ in self.test_loader:
                images = images.reshape(-1, 28 * 28).to(self.device)
                mu, sigma, outputs = self.model(images)
                loss_regularization = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
                loss_reconstruction = self.criterion(outputs, images)
                loss = loss_regularization + loss_reconstruction

            print(f'The loss of the test data is: {loss.item()} %')

    def inference(self, digit, num_examples=1):
        images = []
        idx = 0
        for x, y in self.test_loader:
            if y == idx:
                images.append(x)
                idx += 1
            if idx == 10:
                break

        encodings_digit = []
        for d in range(10):
            with torch.no_grad():
                mu, sigma = self.model.encode(images[d].view(1, 784))
            encodings_digit.append((mu, sigma))

        mu, sigma = encodings_digit[digit]
        for example in range(num_examples):
            epsilon = torch.randn_like(sigma)
            z = mu + sigma * epsilon
            out = self.model.decode(z)
            out = out.view(-1, 1, 28, 28)
            save_image(out, f"generated_{digit}_ex{example}.png")


if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters
    model_parameters = {
        'hidden_size': 128,
        'output_size': 20,
    }

    # Hyper parameters
    hyper_parameters = {
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'clip_grad': True,
    }

    # Dataset
    dataset_name = 'mnist'

    # Training & Testing
    IMG_RECONSTRUCTOR = Reconstructor(model_parameters, hyper_parameters, dataset_name, device)
    IMG_RECONSTRUCTOR.train()
    IMG_RECONSTRUCTOR.test()

    for idx in range(10):
        IMG_RECONSTRUCTOR.inference(idx, num_examples=5)
