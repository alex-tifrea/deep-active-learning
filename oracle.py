import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--n_epoch', type=int, default=100, help="num epochs")
    parser.add_argument('--n_labeled', type=int, default=-1, help="number of labeled training samples")
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

    dataset = get_dataset(args.dataset_name)                   # load dataset
    net = get_net(args.dataset_name, device)                   # load network
    strategy = get_strategy("RandomSampling")(dataset, net)    # load strategy

    n_labeled = dataset.n_pool if args.n_labeled == -1 else args.n_labeled
    # start experiment
    dataset.initialize_labels(n_labeled)
    print(f"number of labeled pool: {n_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool-n_labeled}")
    print(f"number of testing pool: {dataset.n_test}")
    print()

    print("Start fine-tuning")
    strategy.train(args.n_epoch)
    preds = strategy.predict(dataset.get_test_data())
    print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")
