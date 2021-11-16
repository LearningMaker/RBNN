import os
import torchvision.datasets

_DATASETS_MAIN_PATH = 'datasets'
_dataset_path = {
    'cifar10': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR10'),
    'cifar100': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR100'),
    'svhn': os.path.join(_DATASETS_MAIN_PATH, 'SVHN'),
    'stl10': os.path.join(_DATASETS_MAIN_PATH, 'STL10'),
    'mnist': os.path.join(_DATASETS_MAIN_PATH, 'MNIST'),
    'fashionmnist': os.path.join(_DATASETS_MAIN_PATH, 'FASHIONMNIST'),
}


def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True):
    train = (split == 'train')
    if name == 'cifar10':
        return torchvision.datasets.CIFAR10(root=_dataset_path['cifar10'],
                                            train=train,
                                            transform=transform,
                                            target_transform=target_transform,
                                            download=download)
    elif name == 'cifar100':
        return torchvision.datasets.CIFAR100(root=_dataset_path['cifar100'],
                                             train=train,
                                             transform=transform,
                                             target_transform=target_transform,
                                             download=download)
    elif name == 'svhn':
        return torchvision.datasets.SVHN(root=_dataset_path['svhn'],
                                         split=split,
                                         transform=transform,
                                         target_transform=target_transform,
                                         download=download)
    elif name == 'stl10':
        return torchvision.datasets.STL10(root=_dataset_path['stl10'],
                                          split=split,
                                          transform=transform,
                                          target_transform=target_transform,
                                          download=download)
    elif name == 'mnist':
        return torchvision.datasets.MNIST(root=_dataset_path['mnist'],
                                          train=train,
                                          transform=transform,
                                          target_transform=target_transform,
                                          download=download)
    elif name == 'fashionmnist':
        return torchvision.datasets.FashionMNIST(root=_dataset_path['fashionmnist'],
                                                 train=train,
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 download=download)
