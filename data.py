import numpy as np
import os
import torch
from torchvision import datasets
from torch.utils.data import Subset

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)

    def cal_labeled_acc(self, preds, labeled_idxs):
        return 1.0 * (self.Y_train[labeled_idxs] == preds).sum().item() / len(labeled_idxs)

    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test



def get_MNIST(handler, root):
    raw_train = datasets.MNIST(os.path.join(root, 'data/MNIST'), train=True, download=True)
    raw_test = datasets.MNIST(os.path.join(root, 'data/MNIST'), train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_FashionMNIST(handler, root):
    raw_train = datasets.FashionMNIST(os.path.join(root, 'data/FashionMNIST'), train=True, download=True)
    raw_test = datasets.FashionMNIST(os.path.join(root, 'data/FashionMNIST'), train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_SVHN(handler, root):
    data_train = datasets.SVHN(os.path.join(root, 'data/SVHN'), split='train', download=True)
    data_test = datasets.SVHN(os.path.join(root, 'data/SVHN'), split='test', download=True)
    return Data(data_train.data[:40000], torch.from_numpy(data_train.labels)[:40000], data_test.data[:40000], torch.from_numpy(data_test.labels)[:40000], handler)

def get_CIFAR10(handler, root):
    data_train = datasets.CIFAR10(os.path.join(root, 'data/CIFAR10'), train=True, download=True)
    data_test = datasets.CIFAR10(os.path.join(root, 'data/CIFAR10'), train=False, download=True)
    return Data(data_train.data[:40000], torch.LongTensor(data_train.targets)[:40000], data_test.data[:40000], torch.LongTensor(data_test.targets)[:40000], handler)

def get_CIFAR100(handler, root):
    data_train = datasets.CIFAR100(os.path.join(root, 'data/CIFAR100'), train=True, download=True)
    data_test = datasets.CIFAR100(os.path.join(root, 'data/CIFAR100'), train=False, download=True)
    return Data(data_train.data[:40000], torch.LongTensor(data_train.targets)[:40000], data_test.data[:40000], torch.LongTensor(data_test.targets)[:40000], handler)

def get_EuroSAT(handler, root):
    all_data = datasets.EuroSAT(os.path.join(root, 'data/EuroSAT'), download=True)
    total_n = len(all_data)
    idxs = np.random.permutation(total_n)
    images = np.array([np.array(x[0]) for x in all_data])[idxs]
    targets = np.array([x[1] for x in all_data])[idxs]
    n_train = int(total_n * 0.8)
#     data_test = datasets.EuroSAT(os.path.join(root, 'data/EuroSAT'), download=True)
    return Data(images[:n_train], torch.LongTensor(targets)[:n_train], images[n_train:], torch.LongTensor(targets)[n_train:], handler)

def get_PCAM(handler, root):
    MAX_N_PCAM_TRAIN = 50000
    MAX_N_PCAM_TEST = 10000

    data_train = Subset(
        datasets.PCAM(os.path.join(root, 'data/PCAM'), split='train', download=True),
        np.arange(MAX_N_PCAM_TRAIN)
    )
    idxs = np.random.permutation(len(data_train))
    train_images = np.array([np.array(x[0]) for x in data_train])[idxs]
    train_targets = np.array([x[1] for x in data_train])[idxs]

    data_test = Subset(
        datasets.PCAM(os.path.join(root, 'data/PCAM'), split='val', download=True),
        np.arange(MAX_N_PCAM_TEST)
    )
    idxs = np.random.permutation(len(data_test))
    test_images = np.array([np.array(x[0]) for x in data_test])[idxs]
    test_targets = np.array([x[1] for x in data_test])[idxs]

    return Data(train_images, torch.LongTensor(train_targets), test_images,
            torch.LongTensor(test_targets), handler)
