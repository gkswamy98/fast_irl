from filt import FILTER
from bc import BehavioralCloning
from data_utils import fetch_demos
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train expert policies.')
    parser.add_argument('-e', '--env', choices=['walker', 'hopper', 'halfcheetah', 'antmaze', 'bullet-all'], required=True)
    parser.add_argument('-a', '--algo', choices=['mm', 'filter-nr', 'filter-br', 'bc'], required=True)
    parser.add_argument('-s', '--seed', required=True)
    args = parser.parse_args()

    if args.seed is not None and args.seed.isdigit():
        seed = int(args.seed)

    if args.env == 'walker':
        envs = ["Walker2DBulletEnv-v0"]
    elif args.env == "hopper":
        envs = ["HopperBulletEnv-v0"]
    elif args.env == "halfcheetah":
        envs = ["HalfCheetahBulletEnv-v0"]
    elif args.env == 'antmaze':
        envs = ["antmaze-large-diverse-v2", "antmaze-large-play-v2"]
    elif args.env == "bullet-all":
        envs = ["Walker2DBulletEnv-v0", "HopperBulletEnv-v0", "HalfCheetahBulletEnv-v0"]

    for env in envs:
        print(env)
        tup = fetch_demos(env)
        if args.algo == "mm":
            for i in range(seed):
                print(f"SEED {i}")
                flt = FILTER(env)
                flt.train(*tup, n_seed=i, n_exp=25, alpha=0, no_regret=True, device="cuda:0")
        elif args.algo == "filter-nr":
            for i in range(seed):
                print(f"SEED {i}")
                flt = FILTER(env)
                flt.train(*tup, n_seed=i, n_exp=25, alpha=1, no_regret=True, device="cuda:0")
        elif args.algo == "filter-br":
            for i in range(seed):
                print(f"SEED {i}")
                flt = FILTER(env)
                flt.train(*tup, n_seed=i, n_exp=25, alpha=1, no_regret=False, device="cuda:0")
        elif args.algo == "bc":
            for i in range(seed):
                print(f"SEED {i}")
                BehavioralCloning(env, *tup[1:3], n_seed=i, device="cuda:0")