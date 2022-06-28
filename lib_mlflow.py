#!/usr/bin/env python3
import os
import time
import mlflow
import pandas as pd


METRIC_COLUMNS = [
    "acc_at_tnr",
    "ap",
    "approx_tpr",
    "auroc",
    "fpr_at_tpr",
    "tpr_at_tnr",
]

COLUMNS = ["method_name", "id", "ood", *METRIC_COLUMNS]


def setup_mlflow():
    os.environ["MLFLOW_TRACKING_USERNAME"] = "exp-07.mlflow-yang.alex"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "alexpwd"
    remote_server_uri = "https://exp-07.mlflow-yang.inf.ethz.ch"
    mlflow.set_tracking_uri(remote_server_uri)


def assert_df_has_one_experiment_per_scenario(df: pd.DataFrame):
    per_scenario = df.groupby(["method_name", "id", "ood"]).count()
    assert (per_scenario == 1).all().all(), f"Failures in {df}"


def assert_df_sane(df: pd.DataFrame):
    assert set(COLUMNS) == set(df.columns.values.tolist())


def retry(f):
    while True:
        try:
            return f()
        except:
            print("Retrying MLFlow...")
            time.sleep(5)
            continue
