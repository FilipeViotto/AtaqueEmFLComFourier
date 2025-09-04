import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor


def down_cifar():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_set = torchvision.datasets.CIFAR10(root='./data', # Pasta onde os dados serão salvos
        train=True,              # Especifica que é o conjunto de treino
        download=True,           # Baixa se não estiver na pasta 'root'
        transform=transform)     # Aplica as transformações definidas
    
    test_set = torchvision.datasets.CIFAR10(root='./data',
        train=False,             # Especifica que é o conjunto de teste
        download=True,
        transform=transform)
    
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_set, test_set, classes


def down_mnist():
    train_data = datasets.MNIST(
    root = 'data',      # Pasta onde os dados serão salvos
    train = True,       # Especifica que é o conjunto de treino
    download = True,    # Baixa se não estiver na pasta 'root'
    transform = ToTensor()
    )

    test_data = datasets.MNIST(
    root = 'data',
    train = False,      # Especifica que é o conjunto de teste
    download = True,
    transform = ToTensor()
    )

    return train_data, test_data, None

