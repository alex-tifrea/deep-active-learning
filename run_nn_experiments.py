import argparse
import itertools
import os
import sys
import time
import numpy as np
import lib_jobs

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Launches a run that trains an ensemble.")
    parser.add_argument(
        "--goal_tag",
        type=str,
        default="",
        required=False,
        help="Optional goal tag to set.",
    )
    args = parser.parse_args()

    root_dir = "/cluster/scratch/tifreaa/deep_al/"
    log_dir = "/cluster/scratch/tifreaa/deep_al/output_logs/"

    n_init_labeled = 100
    n_query_values = [
        20,
        100,
    ]
    n_round = 50
    n_epoch = 20
    num_repetitions = 10
#     num_repetitions = 1
    strategy_names = [
        "LeastConfidence",
        "RandomSampling",
    ]
    datasets = [
#         "CIFAR10",
#         "SVHN",
#         "CIFAR100",
#         "EuroSAT",
        "PCAM",
    ]


    for config in itertools.product(
        strategy_names,
        datasets,
        n_query_values,
    ):
        (strategy_name, dataset, n_query) = config

        log_file = os.path.join(log_dir, f"log_{np.random.randint(1e10)}")

        base_cli_args = [
            ("root", root_dir),
            ("dataset_name", dataset),
            ("strategy_name", strategy_name),
            ("n_query", n_query),
            ("n_epoch", n_epoch),
            ("n_round", n_round),
            ("n_init_labeled", n_init_labeled),
        ]

        print("Training models in parallel.")
        for model_index in range(num_repetitions):
            cur_log_file = f"{log_file}_model_index_{model_index}"
            cli_args = base_cli_args + [
                ("log_file", cur_log_file),
                ("seed", model_index),
            ]
#             print(cli_args)
            nhours = 4
            mem = 10000
            lib_jobs.launch_bsub(
                nhours=nhours,
                main_python_file="demo.py",
                cli_args=cli_args,
                gin_args=[],
                log_file=cur_log_file,
                need_gpu=True,
                memory_per_cpu=mem,
            )
