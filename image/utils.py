from PIL import Image
from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.datasets import CIFAR10, CIFAR100

from random import sample
import cv2
import numpy as np


class CIFAR10Pair(CIFAR10):
    def __init__(self, root='../data', train=True, transform=None):
        super().__init__(root=root, train=train, transform=transform, download=True)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
            pos_3 = self.transform(img)
            pos_4 = self.transform(img)
            pos_5 = self.transform(img)
            pos_6 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return pos_1, pos_2, pos_3,pos_4, pos_5, pos_6,target


class CIFAR100Pair(CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
            pos_3 = self.transform(img)
            pos_4 = self.transform(img)
            pos_5 = self.transform(img)
            pos_6 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2,pos_3,pos_4,pos_5, pos_6, target


class STL10Pair(STL10):
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
            pos_3 = self.transform(img)
            pos_4 = self.transform(img)
            pos_5 = self.transform(img)
            pos_6 = self.transform(img)

        return pos_1, pos_2,pos_3,pos_4, pos_5, pos_6, target


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.1 * 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


def get_dataset(dataset_name, root='../data', pair=True):
    if pair:
        if dataset_name == 'cifar10':
            train_data = CIFAR10Pair(root=root, train=True, transform=train_transform)
            memory_data = CIFAR10Pair(root=root, train=True, transform=test_transform)
            test_data = CIFAR10Pair(root=root, train=False, transform=test_transform)
        elif dataset_name == 'cifar100':
            train_data = CIFAR100Pair(root=root, train=True, transform=train_transform)
            memory_data = CIFAR100Pair(root=root, train=True, transform=test_transform)
            test_data = CIFAR100Pair(root=root, train=False, transform=test_transform)
        elif dataset_name == 'stl10':
            train_data = STL10Pair(root=root, split='train+unlabeled', transform=train_transform)
            memory_data = STL10Pair(root=root, split='train', transform=test_transform)
            test_data = STL10Pair(root=root, split='test', transform=test_transform)
        else:
            raise Exception('Invalid dataset name')
        return train_data
    else:
        if dataset_name in ['cifar10', 'cifar10_true_label']:
            train_data = CIFAR10(root=root, train=True, transform=train_transform)
            memory_data = CIFAR10(root=root, train=True, transform=test_transform)
            test_data = CIFAR10(root=root, train=False, transform=test_transform)
        elif dataset_name in ['cifar100', 'cifar100_true_label']:
            train_data = CIFAR100(root=root, train=True, transform=train_transform)
            memory_data = CIFAR100(root=root, train=True, transform=test_transform)
            test_data = CIFAR100(root=root, train=False, transform=test_transform)
        elif dataset_name == 'stl10':
            train_data = STL10(root=root, split='train', transform=train_transform)
            memory_data = STL10(root=root, split='train', transform=test_transform)
            test_data = STL10(root=root, split='test', transform=test_transform)
        else:
            raise Exception('Invalid dataset name')

    return train_data, memory_data, test_data

