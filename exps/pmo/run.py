# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from wenhao-gao/mol_opt.
#
# Source:
# https://github.com/wenhao-gao/mol_opt/blob/main/run.py
#
# The license for this can be found in license_thirdparty/LICENSE_PMO.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

from __future__ import print_function
import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))
from tdc import Oracle
from time import time 


def main():
    start_time = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='hparams.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('-s', '--seed', type=int, nargs="+", default=[0])
    parser.add_argument('--task', type=str, default="simple", choices=["simple"])
    parser.add_argument('-o', '--oracles', nargs="+", default=["QED"])
    parser.add_argument('--log_results', action='store_true')
    args = parser.parse_args()
    
    args.method = 'f_rag'

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, "main", args.method)

    sys.path.append(path_main)
    
    print(args.method)
    from main.f_rag.run import f_RAG_Optimizer as Optimizer

    if args.output_dir is None:
        args.output_dir = os.path.join(path_main, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for oracle_name in args.oracles:
        print(f'Optimizing oracle function: {oracle_name}')

        try:
            config = yaml.safe_load(open(args.config))
        except:
            config = yaml.safe_load(open(os.path.join(path_main, args.config)))
        
        oracle = Oracle(name=oracle_name)
        optimizer = Optimizer(args=args)

        for seed in args.seed:
            print('seed', seed)
            optimizer.optimize(oracle=oracle, config=config, seed=seed)

    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))


if __name__ == "__main__":
    main()

