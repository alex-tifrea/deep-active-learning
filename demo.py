from copy import deepcopy
import argparse
import mlflow
from lib_mlflow import retry, setup_mlflow
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=".", help="root")
    parser.add_argument('--log_file', type=str, default="", help="log file")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--n_init_labeled', type=int, default=10000, help="number of init labeled samples")
    parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
    parser.add_argument('--n_epoch', type=int, default=20, help="num epochs")
    parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
    parser.add_argument('--dataset_name', type=str, default="MNIST", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10"], help="dataset")
    parser.add_argument('--strategy_name', type=str, default="RandomSampling",
                        choices=["RandomSampling",
                                 "LeastConfidence",
                                 "MarginSampling",
                                 "EntropySampling",
                                 "LeastConfidenceDropout",
                                 "MarginSamplingDropout",
                                 "EntropySamplingDropout",
                                 "KMeansSampling",
                                 "KCenterGreedy",
                                 "BALDDropout",
                                 "AdversarialBIM",
                                 "AdversarialDeepFool"], help="query strategy")
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
    mlflow.set_experiment("al_vs_pl_nn")

    run_name = f"{args.dataset_name}"
    with mlflow.start_run(run_name=run_name):
        params = deepcopy(vars(args))
        retry(lambda: mlflow.log_params(params))

        dataset = get_dataset(args.dataset_name, args.root)        # load dataset
        net = get_net(args.dataset_name, device)                   # load network
        strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

        # start experiment
        dataset.initialize_labels(args.n_init_labeled)
        print(f"number of labeled pool: {args.n_init_labeled}")
        print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
        print(f"number of testing pool: {dataset.n_test}")
        print()

        retry(lambda: mlflow.log_params({
            "n_unlabeled": dataset.n_pool - args.n_init_labeled,
            "n_test": dataset.n_test,
        }))

        # round 0 accuracy
        print("Round 0")
        strategy.train(n_epoch=args.n_epoch, n_round=0)
        preds = strategy.predict(dataset.get_test_data())
        test_acc = dataset.cal_test_acc(preds)
        print(f"Round 0 testing accuracy: {test_acc}")
        retry(lambda: mlflow.log_metric("test_acc", test_acc, step=0))

        for rd in range(1, args.n_round+1):
            print(f"Round {rd}")

            # query
            query_idxs = strategy.query(args.n_query)

            # update labels
            strategy.update(query_idxs)
            strategy.train(n_epoch=args.n_epoch, n_round=rd)

            # calculate accuracy
            preds = strategy.predict(dataset.get_test_data())
            test_acc = dataset.cal_test_acc(preds)
            print(f"Round {rd} testing accuracy: {test_acc}")
            retry(lambda: mlflow.log_metric("test_acc", test_acc, step=rd))
