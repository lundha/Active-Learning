import time
import subprocess
import argparse
import os


def main(abs_path: str):

    strategies = ["max_entropy", "bayesian_sparse_set", "coreset", "uncertainty"]
    strategies = ["bayesian_sparse_set"]
    processes = [subprocess.Popen(args=["python3", f"{abs_path}/run.py", "--strategy", strategy]) for strategy in strategies]

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/home/martlh/masteroppgave/core-set", help="Abs path to main run file")
    args = parser.parse_args()
    main(abs_path=args.path)
