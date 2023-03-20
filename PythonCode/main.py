import argparse

from utils import import_attr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="", help="")
    
    args, unknown_args = parser.parse_known_args()
    return args


def main(args):
    exp = args.exp
    run_exp = import_attr("experiments.{}.run".format(exp))

    config = {}

    run_exp(config)


if __name__ == "__main__":
    args = parse_args()
    main(args)
