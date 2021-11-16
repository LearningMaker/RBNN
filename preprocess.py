import torchvision.transforms

normalize = {'mean': [0.485, 0.456, 0.406],
             'std': [0.229, 0.224, 0.225]}

gray_normalize = {'mean': [0.5], 'std': [0.5]}


def train_transforms(input_size, gray=False):
    t_list = [torchvision.transforms.Resize(input_size),
              torchvision.transforms.RandomCrop(input_size, padding=4),
              torchvision.transforms.RandomHorizontalFlip()]

    t_list += [torchvision.transforms.Grayscale(3),
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize(**gray_normalize)] if gray else [torchvision.transforms.ToTensor(),
                                                                                 torchvision.transforms.Normalize(**normalize)]
    return torchvision.transforms.Compose(t_list)


def test_transforms(input_size, gray=False):
    t_list = [torchvision.transforms.Resize(input_size)]

    t_list += [torchvision.transforms.Grayscale(3),
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize(**gray_normalize)] if gray else [torchvision.transforms.ToTensor(),
                                                                                 torchvision.transforms.Normalize(
                                                                                     **normalize)]
    return torchvision.transforms.Compose(t_list)


def get_transform(name, input_size, augment):
    if 'cifar' in name or name == 'svhn' or name == 'stl10':
        if augment:
            return train_transforms(input_size)
        else:
            return test_transforms(input_size)
    elif 'mnist' in name:
        if augment:
            return train_transforms(input_size, gray=True)
        else:
            return test_transforms(input_size, gray=True)