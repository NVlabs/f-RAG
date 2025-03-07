# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for f-RAG. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
from tdc import Oracle, Evaluator
from main.optimizer import top_auc

import sys
sys.path.append('../..')
from fusion.utils_eval import calculate_similarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    file = parser.parse_args().file

    evaluator = Evaluator('diversity')
    sa_scorer = Oracle('sa')

    df = pd.read_csv(file, names=['smiles', 'score'])
    df = df.drop_duplicates(subset=['smiles']).iloc[:10000]
    df = df.sort_values(by='score', ascending=False)

    print(f"Avg. Top-1:\t\t{df.iloc[0]['score']}")
    print(f"Avg. Top-10:\t\t{df.iloc[:10]['score'].mean()}")
    print(f"Avg. Top-100:\t\t{df.iloc[:100]['score'].mean()}")

    mol_dict = {df.iloc[i]['smiles']: [df.iloc[i]['score'], df.index[i]] for i in range(len(df))}
    print(f"AUC Top-1:\t\t{top_auc(mol_dict, 1, True, 100, 10000)}")
    print(f"AUC Top-10:\t\t{top_auc(mol_dict, 10, True, 100, 10000):.3f}")
    print(f"AUC Top-100:\t\t{top_auc(mol_dict, 100, True, 100, 10000):.3f}")

    df = df.iloc[:100]
    print(f"Top-100 Diversity:\t{evaluator(df['smiles']):.3f}")
    print(f"Top-100 SA Score:\t{np.mean(sa_scorer(df['smiles'].tolist()))}")

    calculate_similarity(df)
    df = df[df['SIM'] < 0.4]
    print(f"Top-100 Novelty:\t{len(df) / 100}")
