import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

RESNET_INPUT_SIZE = 224
transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(RESNET_INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(RESNET_INPUT_SIZE),
            transforms.CenterCrop(RESNET_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

class MNIST_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x.numpy(), mode="L")
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class SVHN_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms["train"]
        # self.transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        #         ),
        #     ]
        # )

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(np.transpose(x, (1, 2, 0)))
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class CIFAR10_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms["train"]
        # self.transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        #         ),
        #     ]
        # )

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
