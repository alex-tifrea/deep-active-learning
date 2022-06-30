import os
from pathlib import Path
from typing import List, Tuple
import subprocess


def list2cmd(l):
    return " ".join(l)


def form_python_args(
    main_python_file: str,
    cli_args: List[Tuple[str, str]],
    gin_args: List[Tuple[str, str]],
):
    python_args = ["python3", main_python_file]

    for kv_pair in cli_args:
        if len(kv_pair) == 2:
            k, v = kv_pair
            python_args.append(f"--{k}")
            python_args.append(str(v))
        elif len(kv_pair) == 1:
            k = kv_pair[0]
            python_args.append(f"--{k}")
        else:
            raise RuntimeError(f"Arguments can be a singleton or a pair, instead were {kv_pair}.")

    for k, v in gin_args:
        if isinstance(v, str):
            v = '\\"{}\\"'.format(v)
        python_args.append("--gin_param")
        python_args.append(f'"{k}={v}"')
    return python_args


def launch_bsub(
    nhours: int,
    main_python_file: str,
    cli_args: List[Tuple[str, str]],
    gin_args: List[Tuple[str, str]],
    log_file: str = None,
    job_name: str = None,
    memory_per_cpu: int = 4096,
    need_gpu: bool = True,
    tesla: bool = False,
):
    # fmt: off
    bsub_args = [
        "bsub",
        "-W", f"{nhours}:00",
        "-n", str(5),
        "-R", f"\"rusage[mem={memory_per_cpu},scratch=4000]\"",
    ]
    if need_gpu:
        bsub_args += [
            "-R", "\"rusage[ngpus_excl_p=1]\"",
            "-R", "\"select[gpu_mtotal0>=10240]\"",
        ]
    # fmt: on

    if tesla:
        print("Requesting a tesla GPU")
        bsub_args.extend(["-R", '"select[gpu_model0==TeslaV100_SXM2_32GB]"'])

    if job_name is not None:
        bsub_args.extend(["-J", job_name])

    if log_file is not None:
        bsub_args.extend(["-o", log_file])

    python_args = form_python_args(main_python_file, cli_args, gin_args)

    python_command = list2cmd(python_args)
    bsub_command = list2cmd(bsub_args + [python_command])

    subprocess.check_call(bsub_command, env=os.environ, shell=True)


def launch_local(
    main_python_file: str,
    cli_args: List[Tuple[str, str]],
    gin_args: List[Tuple[str, str]],
):
    python_args = form_python_args(main_python_file, cli_args, gin_args)
    python_args = [a.replace('"', "").replace("\\", '"') for a in python_args]
    subprocess.check_call(python_args, env=os.environ)
