import argparse
from copy import deepcopy
import mlflow
from lib_mlflow import retry, setup_mlflow
import numpy as np
import os
import torch
import time
from utils import get_dataset, get_net, get_strategy
from pprint import pprint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=".", help="root")
    parser.add_argument('--ckpt_root', type=str, default=".", help="checkpoint root")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--n_epoch', type=int, default=100, help="num epochs")
    parser.add_argument('--n_labeled', type=int, default=-1, help="number of labeled training samples")
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

    setup_mlflow()
    mlflow.set_experiment("finetuned_nn")

    run_name = f"{args.dataset_name}"
    with mlflow.start_run(run_name=run_name):
        params = deepcopy(vars(args))
        del params["n_labeled"]
        del params["ckpt_root"]
        retry(lambda: mlflow.log_params(params))

        dataset = get_dataset(args.dataset_name, args.root)        # load dataset
        net = get_net(args.dataset_name, device)                   # load network
        strategy = get_strategy("RandomSampling")(dataset, net)    # load strategy

        n_labeled = dataset.n_pool if args.n_labeled == -1 else args.n_labeled
        # start experiment
        dataset.initialize_labels(n_labeled)
        print(f"number of labeled pool: {n_labeled}")
        print(f"number of unlabeled pool: {dataset.n_pool-n_labeled}")
        print(f"number of testing pool: {dataset.n_test}")
        print()

#         np.random.seed(time.time())
        ckpt_root = os.path.join(args.ckpt_root, "ckpt/", f"{args.dataset_name}_{np.random.randint(1e10)}")
        os.makedirs(ckpt_root, exist_ok=True)

        retry(lambda: mlflow.log_params({
            "n_labeled": n_labeled,
            "n_unlabeled": dataset.n_pool - n_labeled,
            "n_test": dataset.n_test,
            "ckpt_root": ckpt_root,
        }))

        print("Start fine-tuning")
        strategy.train(n_epoch=args.n_epoch, ckpt_root=ckpt_root)

        preds = strategy.predict(dataset.get_test_data())
        print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")
