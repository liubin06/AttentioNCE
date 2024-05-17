
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import cv2
import numpy as np
from typing import Any, Callable,  Optional

def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



class tinyImagePair(ImageFolder):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any] = pil_loader,
            transform: Optional[Callable] = None,
    ) -> None:
        super(tinyImagePair, self).__init__(root, transform=transform)
        self.loader = loader

    def __getitem__(self, index):
        img_id, target = self.imgs[index]
        img = self.loader(img_id)
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
            pos_3 = self.transform(img)
            pos_4 = self.transform(img)
            pos_5 = self.transform(img)
            pos_6 = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2,pos_3, pos_4, pos_5,pos_6, target

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
    GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])


def get_dataset(pair=True):
    if pair:
        train_data = tinyImagePair(root='../tiny-imagenet-200/train',  transform=train_transform)
        return train_data

    else:
        train_data = ImageFolder(root='../tiny-imagenet-200/train', transform=train_transform)
        memory_data = ImageFolder(root='../tiny-imagenet-200/train', transform=test_transform)
        test_data = ImageFolder(root='../tiny-imagenet-200/val', transform=test_transform)
        return train_data,memory_data,test_data

