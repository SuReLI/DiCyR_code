from PIL import Image
from torch.utils.data import Dataset
import os

import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, USPS, SVHN
import torchvision

import numpy as np
import random
import h5py
import cv2

def random_affine_augmentation(x):
    M = np.float32([[1 + np.random.normal(0.0, 0.1), np.random.normal(0.0, 0.1), 0], 
                    [np.random.normal(0.0, 0.1), 1 + np.random.normal(0.0, 0.1), 0]])
    rows, cols = x.shape[1:3]
    dst = cv2.warpAffine(x.numpy()[0], M, (cols,rows))
    dst = np.expand_dims(dst, axis=0)
    return torch.from_numpy(dst)

def gaussian_blur(x, sigma=0.1):
	ksize = int(sigma + 0.5) * 8 + 1
	dst = cv2.GaussianBlur(x.numpy(), (ksize, ksize), sigma)
	return torch.from_numpy(dst)

def color_random_affine_augmentation(x):
	M = np.float32([[1 + np.random.normal(0.0, 0.1), np.random.normal(0.0, 0.1), 0], 
				[np.random.normal(0.0, 0.1), 1 + np.random.normal(0.0, 0.1), 0]])
	rows, cols = x.shape[1:3]
	dst = cv2.warpAffine(np.transpose(x.numpy(), [1, 2, 0]), M, (cols,rows))
	dst = np.transpose(dst, [2, 0, 1])
	return torch.from_numpy(dst)

class MNIST_M(Dataset):
    """MNIST_M Dataset."""
    def __init__(self, data_root, data_list, transform=None):
        """Initialize MNIST_M data set.
        Keyword Params:
            root -- the root folder containing the data
            data_list -- the file containing list of images - labels
            transform (optional) -- tranfrom images when loading
        """
        self.root = data_root
        self.transform = transform
        with open(data_list, 'r') as f:
            self.data_list = f.readlines()

    def __len__(self):
        """Get length of dataset."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Get item."""
        img_name, labels = self.data_list[idx].split()
        imgs = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        if self.transform:
            imgs = self.transform(imgs)

        labels = int(labels)

        return imgs, labels


def load_mnist_m(**kwargs):
    """Load MNIST_M dataloader.
    :**kwargs: arguments to pass to dataloader constructor
    :returns: train_loader, test_loader
    """
    # Load target images
    img_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    trainset = MNIST_M(
        data_root=os.path.join('../data', 'mnist_m', 'mnist_m_train'),
        data_list=os.path.join('../data', 'mnist_m', 'mnist_m_train_labels.txt'),
        transform=img_transform
    )

    testset = MNIST_M(
        data_root=os.path.join('../data', 'mnist_m', 'mnist_m_test'),
        data_list=os.path.join('../data', 'mnist_m', 'mnist_m_test_labels.txt'),
        transform=img_transform
    )
    return get_loader(trainset, **kwargs), get_loader(testset, **kwargs)


def get_loader(dataset, **kwargs):
    """Get dataloader from dataset."""
    return torch.utils.data.DataLoader(dataset, **kwargs)


def load_mnist(img_size=28, augment=False, **kwargs):
    transformations = [transforms.Resize(img_size)]
    transformations.append(transforms.ToTensor())
    if augment:
        transformations.append(transforms.Lambda(lambda x: random_affine_augmentation(x)))
        transformations.append(transforms.Lambda(lambda x: gaussian_blur(x)))
    img_transform = transforms.Compose(transformations)
    test_transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    train_set = MNIST('../data', transform=img_transform, download=True, train=True)
    test_set = MNIST('../data', transform=test_transform, download=True, train=False)
    return get_loader(train_set, **kwargs), get_loader(test_set, **kwargs)


def load_usps(img_size=28, augment=False, **kwargs):
    transformations = [transforms.Resize(img_size)]
    transformations.append(transforms.ToTensor())
    if augment:
        transformations.append(transforms.Lambda(lambda x: random_affine_augmentation(x)))
        transformations.append(transforms.Lambda(lambda x: gaussian_blur(x)))
    img_transform = transforms.Compose(transformations)
    test_transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    train_set = USPS('../data', transform=img_transform, download=True)
    test_set = USPS('../data', transform=test_transform, download=True)
    return get_loader(train_set, **kwargs), get_loader(test_set, **kwargs)


def load_svhn(img_size=(32, 32), rotation=0, grayscale=False, split=1000, **kwargs):
    transformations = [transforms.Resize(img_size), transforms.ColorJitter(hue=.05, saturation=.15)]
    if rotation:
        transformations.append((transforms.RandomRotation(rotation)))
    if grayscale:
        transformations.append((transforms.Grayscale()))
    transformations.append(transforms.ToTensor())
    img_transform = transforms.Compose(transformations)

    dataset = SVHN('../data', transform=img_transform, download=True)
    train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset) - split, split])
    return get_loader(train_set, **kwargs), get_loader(test_set, **kwargs)

class Shape(Dataset):
    """3d shape Dataset."""

    def __init__(self, data_path, transform=None, label_idx=4, data_size=20000):
        dataset = h5py.File(data_path, 'r')
        dataset_size = len(dataset['images'])
        indexes = np.sort(random.sample(range(dataset_size), data_size))
        self.images = dataset['images'][indexes]
        self.labels = dataset['labels'][indexes][:, label_idx]
        self.transform = transform
        self.labels_unique = np.unique(self.labels)
        # self.label_idx = label_idx

    def __len__(self):
        """Get length of dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get item."""
        labels = self.labels[idx]
        labels = np.where(self.labels_unique == labels)[0][0]
        imgs = self.images[idx]

        if self.transform:
            imgs = self.transform(imgs)

        return imgs, int(labels)


def load_shape(label_idx=4, data_size=20000, **kwargs):
    # Load target images
    img_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize(32),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    trainset = Shape('../data/3dshapes.h5', transform=img_transform, label_idx=label_idx, data_size=data_size)

    return get_loader(trainset, **kwargs)


class Synsigns(Dataset):
    """Synsigns Dataset."""
    def __init__(self, data_root, data_list, transform=None):
        """Initialize Synsigns data set.
        Keyword Params:
            root -- the root folder containing the data
            data_list -- the file containing list of images - labels
            transform (optional) -- tranfrom images when loading
        """
        self.root = data_root
        self.transform = transform
        with open(data_list, 'r') as f:
            self.data_list = f.readlines()

    def __len__(self):
        """Get length of dataset."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Get item."""
        img_name, labels, _ = self.data_list[idx].split()
        imgs = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        if self.transform:
            imgs = self.transform(imgs)

        labels = int(labels)

        return imgs, labels


def load_synsigns(img_size=32, **kwargs):
    """Load Synsigns dataloader.
    :**kwargs: arguments to pass to dataloader constructor
    :returns: loader
    """
    # Load target images
    img_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    trainset = Synsigns(
        data_root=os.path.join('../data/synsigns/synthetic_data/'),
        data_list=os.path.join('../data/synsigns/synthetic_data/', 'train_labelling.txt'),
        transform=img_transform
    )

    
    return get_loader(trainset, **kwargs)


def load_GTSRB(img_size=32,**kwargs):
    """Load GTSRB dataloader.
    :**kwargs: arguments to pass to dataloader constructor
    :returns: loader
    """
    data_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)), # spatial size of vgg-f input,
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    data = torchvision.datasets.ImageFolder(root='../data/GTSRB/Final_Training/Images/', transform=data_transform)
    loader = torch.utils.data.DataLoader(data, **kwargs)
    
    return loader

    return get_loader(trainset, **kwargs)
