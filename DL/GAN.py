import torch
import torch.nn as nn
from torch import optim
from datasets import load_classification_data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, leaky_relu_slope):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU(leaky_relu_slope)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

        self.initialize_weights()

    def forward(self, x):
        out = self.fc2(self.relu(self.fc1(x)))
        # To ensure the output is between (-1, 1)
        return self.tanh(out)

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


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, leaky_relu_slope):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU(leaky_relu_slope)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigm = nn.Sigmoid()

        self.initialize_weights()

    def forward(self, x):
        out = self.fc2(self.relu(self.fc1(x)))
        # To ensure the output is between (0, 1)
        return self.sigm(out)

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
class IMG_Generator:
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
        image_size = 28 * 28
        self.discriminator = Discriminator(
            image_size, model_params['discriminator_hidden'], model_params['leaky_relu_slope']
        ).to(device)
        self.noise_dim = model_params['noise_dim']
        self.generator = Generator(
            self.noise_dim, model_params['discriminator_hidden'], image_size, model_params['leaky_relu_slope']
        ).to(device)

        # Loss
        self.criterion = nn.BCELoss()

        # Optimizer
        learning_rate = hyper_params['learning_rate']
        self.optimizer_dis = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=learning_rate)

        # Scaler: helps to reduce RAM and train faster
        self.scaler = torch.cuda.amp.GradScaler()

        # Scheduler
        # When a metric stopped improving for 'patience' number of epochs, the learning rate is reduced by a factor of 2-10.
        self.scheduler_dis = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_dis, patience=0.1*hyper_params['num_epochs'], factor=0.5, verbose=True)
        self.scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, patience=0.1*hyper_params['num_epochs'], factor=0.5, verbose=True)
        # # Reduce the learning rate every num_epochs/10 by 0.75
        # self.scheduler_dis = optim.lr_scheduler.StepLR(self.optimizer_dis, step_size=hyper_params['num_epochs'], gamma=0.75, verbose=True)
        # self.scheduler_gen = optim.lr_scheduler.StepLR(self.optimizer_gen, step_size=hyper_params['num_epochs'], gamma=0.75, verbose=True)

        # Other parameters
        self.num_epochs = hyper_params['num_epochs']
        self.clip_grad = hyper_params['clip_grad']
        self.device = device

        # Tensorboard
        self.writer_real = SummaryWriter(f"runs/{dataset_name}_GAN/real")
        self.writer_fake = SummaryWriter(f"runs/{dataset_name}_GAN/fake")

    def train(self):
        tensorboard_step = 0
        n_total_steps = len(self.train_loader)
        for epoch in range(self.num_epochs):
            # # If loading from a checkpoint is being carried out
            # # 1. Create the model and optimizer
            # # 2. Load the state_dict for each
            # # Since they are part of ImageClassifier, step 1 is skipped
            # checkpoint = torch.load("./checkpoints/classifier_CNN.pth")
            # self.model.load_state_dict(checkpoint["model_state"])
            # self.optimizer.load_state_dict(checkpoint["model_state"])
            losses_dis = []
            losses_gen = []
            for i, (images, _) in enumerate(self.train_loader):
                # Convert it to proper size: [n_batch, 784]
                images = images.reshape(-1, 28 * 28).to(self.device)
                # images = images.view(images.shape[0], self.input_size).to(self.device)

                # Forward pass
                if str(self.device) == "cpu":
                    # Compute the discriminator objective: max log(D(real)) + log(1 - D(G(z)))
                    dis_output_images = self.discriminator(images).view(-1)
                    dis_loss_images = self.criterion(dis_output_images, torch.ones_like(dis_output_images))
                    batch_size = images.shape[0]
                    noise = torch.randn(batch_size, self.noise_dim).to(device)
                    fake_images = self.generator(noise)
                    dis_output_fakes = self.discriminator(fake_images).view(-1)
                    dis_loss_fakes = self.criterion(dis_output_fakes, torch.ones_like(dis_output_fakes))
                    loss_dis = 0.5 * (dis_loss_images + dis_loss_fakes)
                    losses_dis.append(loss_dis.item())

                    # Backward and optimize
                    self.optimizer_dis.zero_grad()
                    loss_dis.backward(retain_graph=True)
                    if self.clip_grad:
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1)
                    self.optimizer_dis.step()

                    # Compute the generator objective: max log(D(G(z)))
                    dis_output_fakes = self.discriminator(fake_images).view(-1)
                    loss_gen = self.criterion(dis_output_fakes, torch.ones_like(dis_output_fakes))
                    losses_gen.append(loss_gen.item())

                    # Backward and optimize
                    self.optimizer_gen.zero_grad()
                    loss_gen.backward()
                    if self.clip_grad:
                        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1)
                    self.optimizer_gen.step()

                else:

                    with torch.cuda.amp.autocast():
                        # Compute the discriminator objective: max log(D(real)) + log(1 - D(G(z)))
                        dis_output_images = self.discriminator(images).view(-1)
                        dis_loss_images = self.criterion(dis_output_images, torch.ones_like(dis_output_images))
                        batch_size = images.shape[0]
                        noise = torch.randn(batch_size, self.noise_dim).to(device)
                        fake_images = self.generator(noise)
                        dis_output_fakes = self.discriminator(fake_images).view(-1)
                        dis_loss_fakes = self.criterion(dis_output_fakes, torch.ones_like(dis_output_fakes))
                        loss_dis = 0.5 * (dis_loss_images + dis_loss_fakes)
                        losses_dis.append(loss_dis.item())

                        # Backward and optimize
                        self.optimizer_dis.zero_grad()
                        self.scaler.scale(loss_dis).backward()
                        if self.clip_grad:
                            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1)
                        self.scaler.step(self.optimizer_dis)
                        self.scaler.update()

                        # Compute the generator objective: max log(D(G(z)))
                        dis_output_fakes = self.discriminator(fake_images).view(-1)
                        loss_gen = self.criterion(dis_output_fakes, torch.ones_like(dis_output_fakes))
                        losses_gen.append(loss_gen.item())

                        # Backward and optimize
                        self.optimizer_gen.zero_grad()
                        self.scaler.scale(loss_gen).backward()
                        if self.clip_grad:
                            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1)
                        self.scaler.step(self.optimizer_gen)
                        self.scaler.update()

                # Print every 100 optimizer steps
                if (i + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{n_total_steps}], Dis/Gen Losses: {loss_dis.item():.4f}, {loss_gen.item():.4f}')

                    # # Tensorboard Plots
                    # with torch.no_grad():
                    #     noise = torch.randn((batch_size, self.noise_dim)).to(device)
                    #     fake = self.generator(noise).reshape(-1, 1, 28, 28)
                    #     data = images.reshape(-1, 1, 28, 28)
                    #     img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    #     img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                    #
                    #     self.writer_fake.add_image(
                    #         "Mnist Fake Images", img_grid_fake, global_step=tensorboard_step
                    #     )
                    #     self.writer_real.add_image(
                    #         "Mnist Real Images", img_grid_real, global_step=tensorboard_step
                    #     )
                    #     tensorboard_step += 1

            mean_loss_dis = sum(losses_dis) / len(losses_dis)
            self.scheduler_dis.step(metrics=mean_loss_dis)

            mean_loss_gen = sum(losses_gen) / len(losses_gen)
            self.scheduler_gen.step(metrics=mean_loss_gen)

            # Two ways to save the models
            torch.save(self.discriminator.state_dict(), "models/discriminator_GAN.pth")
            torch.save(self.generator.state_dict(), "models/generator_GAN.pth")
            # torch.save(self.models, "./models/discriminator_GAN.pth")
            # torch.save(self.models, "./models/generator_GAN.pth")

    def test(self):
        # # If a saved models is being tested either use
        # # 1. Create the model and load the state_dict
        # # Since the model is part of ImageClassifier, we just load the state_dict
        self.discriminator.load_state_dict(torch.load("models/discriminator_GAN.pth"))
        self.generator.load_state_dict(torch.load("models/generator_GAN.pth"))
        # # 2. Load the whole model
        # self.discriminator = torch.load("./models/discriminator_GAN.pth")
        # self.generator = torch.load("./models/generator_GAN.pth")
        self.discriminator.eval()
        self.generator.eval()
        with torch.no_grad():
            for images, _ in self.test_loader:
                # Convert it to proper size: [n_batch, 784]
                images = images.reshape(-1, 28 * 28).to(self.device)
                # images = images.view(images.shape[0], self.input_size).to(self.device)

                # Compute the discriminator objective: max log(D(real)) + log(1 - D(G(z)))
                dis_output_images = self.discriminator(images).view(-1)
                dis_loss_images = self.criterion(dis_output_images, torch.ones_like(dis_output_images))
                batch_size = images.shape[0]
                noise = torch.randn(batch_size, self.noise_dim).to(device)
                fake_images = self.generator(noise)
                dis_output_fakes = self.discriminator(fake_images).view(-1)
                dis_loss_fakes = self.criterion(dis_output_fakes, torch.ones_like(dis_output_fakes))
                loss_dis = 0.5 * (dis_loss_images + dis_loss_fakes)

                # Compute the generator objective: max log(D(G(z)))
                dis_output_fakes = self.discriminator(fake_images).view(-1)
                loss_gen = self.criterion(dis_output_fakes, torch.ones_like(dis_output_fakes))

            print(f'Test Results *** Dis/Gen Losses: {loss_dis.item():.4f}, {loss_gen.item():.4f}')


# Running the code
if __name__ == "__main__":

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters
    model_parameters = {
        'noise_dim': 64,
        'generator_hidden': 128,
        'discriminator_hidden': 128,
        'leaky_relu_slope': 0.1,
    }

    # Hyper parameters
    hyper_parameters = {
        'num_epochs': 50,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'clip_grad': True,
    }

    # Dataset
    dataset_name = 'mnist'

    # Training & Testing
    IMG_GENERATOR = IMG_Generator(model_parameters, hyper_parameters, dataset_name, device)
    IMG_GENERATOR.train()
    IMG_GENERATOR.test()
