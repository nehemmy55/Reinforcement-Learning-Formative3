import argparse
import json

from train import parse_args, train

def config_to_argv(member, config, total_steps):
    """Turn {"batch_size": 32, ...} into ["--batch-size", "32", ...] for train.py"""
    argv = ["--member", member]
    for key, value in config.items():
        argv += [f"--{key.replace('_', "-")}", str(value)]
    if total_steps is not None:
        argv += ["--total-steps", str(total_steps)]
    return argv

def main():
    p = argparse.ArgumentParser(description="Run a member's DQN experiments.")
    p.add_argument("--member", required=True, help="reads experiments/<member>.json")
    p.add_argument("--only", type=int, nargs="+", help="experiment numbers to run (default: all)")
    p.add_argument("--total-steps", type=int, help="override training length for every experiment")
    args = p.parse_args()

    with open(f"experiments/{args.member}.json") as f:
        experiments = json.load(f)

    for config in experiments:
        if args.only and config["exp"] not in args.only:
            continue
        train(parse_args(config_to_argv(args.member, config, args.total_steps)))


if __name__ == "__main__":
    main()