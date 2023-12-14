from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms


def load_classification_data(data_name, data_info=True):
    train_dataset, test_dataset, class_labels = None, None, None
    if data_name == 'mnist':
        # MNIST dataset
        # Transform the dataset to Tensors of normalized range [-1, 1]
        # transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.1307,), (0.1307,))])
        transform = transforms.ToTensor()
        train_dataset = MNIST(root='./data',
                              train=True,
                              transform=transform,
                              download=True)
        test_dataset = MNIST(root='./data',
                             train=False,
                             transform=transform)
        class_labels = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    if data_name == 'cifar10':
        # CIFAR10 dataset
        # Transform the dataset to Tensors of normalized range [-1, 1]
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = CIFAR10(root='./data',
                                train=True,
                                transform=transform,
                                download=True)
        test_dataset = CIFAR10(root='./data',
                               train=False,
                               transform=transform)
        class_labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_dataset, test_dataset, class_labels
