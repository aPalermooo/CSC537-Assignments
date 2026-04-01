##############################################################################
# Name:           unpacker.py
# Author:         Xander Palermo <ajp2s@missouristate.edu>
# Description:
# Date:           3 April 2026
#
# Class:          CSC 537: Deep Learning
# Professor:      Mukulika Ghosh
# Assignment:     Assignment 3
##############################################################################
import os
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import v2


def unpickle(file):  # given unpacking method <https://www.cs.toronto.edu/~kriz/cifar.html>
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# Global Vars
metadata = unpickle('cifar-10-batches-py/batches.meta')
classes = [c.decode('utf8') for c in metadata[b'label_names']]

TRAINING_SIZE = 15000
VALIDATION_SIZE = 10000
TESTING_SIZE = 10000



def unpack() -> None:


    # Metadata
    ## Check classification and verify directories

    for c in classes:
        os.makedirs('./dataset', exist_ok=True)
        os.makedirs('./dataset/complete', exist_ok=True)
        os.makedirs('./dataset/complete/training', exist_ok=True)
        os.makedirs('./dataset/complete/training/' + c, exist_ok=True)
        os.makedirs('./dataset/complete/testing', exist_ok=True)
        os.makedirs('./dataset/complete/testing/' + c, exist_ok=True)

    ## Images
    ## Unpickle each binary file and save the content as images (according to classification)

    for index in range(1, 6):
        f_name = f'cifar-10-batches-py/data_batch_{index}'
        d = unpickle(f_name)


        images = d[b'data']
        labels = d[b'labels']
        for i, (image, label) in enumerate(zip(images, labels)):
            image = image.reshape(3,32,32).transpose((1,2,0))
            label = classes[label]
            Image.fromarray(image).save(f'./dataset/complete/training/{label}/{i}.png')

    f_name = f'cifar-10-batches-py/test_batch'
    d = unpickle(f_name)

    ## Repeat for testing dataset
    images = d[b'data']
    labels = d[b'labels']
    for i, (image, label) in enumerate(zip(images, labels)):
        image = image.reshape(3,32,32).transpose((1,2,0))
        label = classes[label]
        Image.fromarray(image).save(f'./dataset/complete/testing/{label}/{i}.png')


# pack Dataset

mean = torch.zeros(3)
std = torch.zeros(3)
n_samples = 0

raw_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

raw_dataset = datasets.ImageFolder('./dataset/complete/training', transform=raw_transform)
loader = DataLoader(raw_dataset, batch_size=64, shuffle=False)

for images, _ in loader:
    batch_size = images.size(0)

    images = images.view(batch_size, 3, -1)

    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    n_samples += batch_size

mean /= n_samples
std /= n_samples

train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.RandomCrop(size=32, padding=4),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean.tolist(), std=std.tolist()),
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean.tolist(), std=std.tolist()),
])

full_training = datasets.ImageFolder('./dataset/complete/training', transform=train_transform)
full_testing  = datasets.ImageFolder('./dataset/complete/testing',  transform=test_transform)

class_indices = defaultdict(list)
for idx, (_, label) in enumerate(full_training.samples):
    class_indices[label].append(idx)

train_indices = []
val_indices   = []

num_samples_train =     TRAINING_SIZE   // len(classes)
num_samples_validate =  VALIDATION_SIZE // len(classes)
num_samples_test =      TESTING_SIZE    // len(classes)

for indices in class_indices.values():
    train_indices.extend(indices[:num_samples_train])
    val_indices.extend(indices[num_samples_train:num_samples_train + num_samples_validate])

training   = Subset(full_training, train_indices)
validation = Subset(full_training, val_indices)
testing    = Subset(full_testing,  range(TESTING_SIZE))



if __name__ == '__main__':
    print("hello world")
