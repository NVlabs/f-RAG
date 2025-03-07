# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for f-RAG. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import pandas as pd

import sys
sys.path.append('../..')
from fusion.utils_eval import calculate_similarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    file = parser.parse_args().file

    df = pd.read_csv(file, names=['smiles', 'DS', 'QED', 'SA', 'TOTAL']).iloc[:3000]
    n_total_smi = len(df)
    df = df.drop_duplicates(subset=['smiles']).iloc[:10000]

    calculate_similarity(df)

    df = df[df['QED'] > 0.5]
    df = df[df['SA'] > 5 / 9]
    df = df[df['SIM'] < 0.4]
    df = df.sort_values(by='DS', ascending=False)
    print(f"Novel Top 5% DS: {df.iloc[:int(n_total_smi * 0.05)]['DS'].mean()}")
