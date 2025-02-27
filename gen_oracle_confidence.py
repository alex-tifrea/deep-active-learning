import os
from copy import deepcopy
import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=".", help="root")
    parser.add_argument('--ckpt_path', type=str, default="", help="ckpt path")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--dataset_name', type=str, default="MNIST",
            choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "CIFAR100",
            "EuroSAT", "PCAM"], help="dataset")
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
    train_pred_probs = strategy.predict_prob(train_data)
    confidences, train_preds = train_pred_probs.max(1)
    train_acc = dataset.cal_labeled_acc(train_preds, np.arange(train_data.Y.shape[0]))

    print("Train acc", train_acc)
    print(train_preds[:5])
    print()
    print(train_pred_probs[:5])
    print()
    print(confidences.shape)
    print(confidences[confidences < 1.0].shape)

    with open(os.path.join(args.root, f'data/{args.dataset_name}/confidences.pkl'), 'wb') as f:
        pickle.dump(confidences, f)
