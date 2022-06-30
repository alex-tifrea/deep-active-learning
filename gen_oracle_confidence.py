from copy import deepcopy
import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=".", help="root")
    parser.add_argument('--ckpt_path', type=str, default="", help="ckpt path")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--dataset_name', type=str, default="MNIST", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10"], help="dataset")
    args = parser.parse_args()
    pprint(vars(args))
    print()

    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = get_dataset(args.dataset_name, args.root)        # load dataset
    net = get_net(args.dataset_name, device)                   # load network
    strategy = get_strategy("RandomSampling")(dataset, net)    # load strategy

    net.load_ckpt(args.ckpt_path)

    print("Generate confidences")
    _, train_data = dataset.get_train_data()
    # train_preds = strategy.predict(train_data)
    # train_acc = dataset.cal_labeled_acc(train_preds, np.arange(train_data.shape[0]))

    # TODO: save in a file for this dataset