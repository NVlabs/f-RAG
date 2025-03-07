# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for f-RAG. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import torch
from rdkit import DataStructs, Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def calculate_similarity(df):
    if not os.path.exists(os.path.join(ROOT_DIR, 'data/zinc250k_novelty.pt')):
        import json
        import pandas as pd
        print('Preprocessing ZINC250k for novelty calculation...')

        df = pd.read_csv(os.path.join(ROOT_DIR, 'data/zinc250k.csv'))
        with open(os.path.join(ROOT_DIR, 'data/valid_idx_zinc250k.json')) as f:
            test_idx = set(json.load(f))
        train_idx = [i for i in range(len(df)) if i not in test_idx]

        train_smiles = df.iloc[train_idx]['smiles']
        train_mols = [Chem.MolFromSmiles(smi) for smi in train_smiles]
        train_smiles = set([Chem.MolToSmiles(mol, isomericSmiles=False) for mol in train_mols])
        train_fps = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in train_mols]    
        torch.save((train_smiles, train_fps), os.path.join(ROOT_DIR, 'data/zinc250k_novelty.pt'))
        print('Novelty preprocessing done')
        
    train_smiles, train_fps = torch.load(os.path.join(ROOT_DIR, 'data/zinc250k_novelty.pt'))
    
    if 'MOL' not in df:
        df['MOL'] = df['smiles'].apply(Chem.MolFromSmiles)
    
    if 'FPS' not in df:
        df['FPS'] = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in df['MOL']]
    
    max_sims = []
    for fps in df['FPS']:
        max_sim = max(DataStructs.BulkTanimotoSimilarity(fps, train_fps))
        max_sims.append(max_sim)
    df['SIM'] = max_sims
