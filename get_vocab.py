# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for f-RAG. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import re
import argparse
import numpy as np
from collections import defaultdict
from tqdm import trange
import pandas as pd
import datamol as dm
import safe as sf
from tdc import Oracle
from rdkit import RDLogger
from fusion.slicer import MolSlicer
RDLogger.DisableLog('rdApp.*')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', choices=['pmo', 'dock'])
    args = parser.parse_args()

    if args.exp == 'pmo':
        props = ['albuterol_similarity',
                 'amlodipine_mpo',
                 'celecoxib_rediscovery',
                 'deco_hop',
                 'drd2',
                 'fexofenadine_mpo',
                 'gsk3b',
                 'isomers_c7h8n2o2',
                 'isomers_c9h10n2o2pf2cl',
                 'jnk3',
                 'median1',
                 'median2',
                 'mestranol_similarity',
                 'osimertinib_mpo',
                 'perindopril_mpo',
                 'qed',
                 'ranolazine_mpo',
                 'scaffold_hop',
                 'sitagliptin_mpo',
                 'thiothixene_rediscovery',
                 'troglitazone_rediscovery',
                 'valsartan_smarts',
                 'zaleplon_mpo']
    else:
        props = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
    
    df = pd.read_csv('data/zinc250k.csv')

    # calculate properties
    if args.exp == 'pmo':
        for prop in props:
            if prop not in df:
                print(f'Calculating {prop}...')
                df[prop] = Oracle(prop)
            df.to_csv('data/zinc250k.csv', index=False)
    else:
        assert 'sa' in df
        for prop in props:
            assert prop in df
    
    # construct vocabulary
    slicer = MolSlicer(shortest_linker=True)
    
    frag2cnt = defaultdict(int)
    frag2score = {prop: defaultdict(float) for prop in props}
    for i in trange(len(df)):
        try:
            for safe_frag in slicer(df['smiles'].iloc[i]):
                if safe_frag is None:
                    continue
                smiles_frag = sf.decode(dm.to_smiles(safe_frag), remove_dummies=False)
                smiles_frag = re.sub(r'\[\d+\*\]', '[1*]', smiles_frag)
                if smiles_frag.count('*') not in {1, 2}:
                    continue
                
                frag2cnt[smiles_frag] += 1
                for prop in props:
                    if args.exp == 'pmo':
                        score = df[prop].iloc[i]
                    else:
                        score = np.clip(df[prop].iloc[i], 0, 20) / 20 * df['qed'].iloc[i] * df['sa'].iloc[i]
                    frag2score[prop][smiles_frag] += score
                    
        except KeyboardInterrupt:
            quit()
        except:
            continue
    
    if not os.path.exists('vocab'):
        os.mkdir('vocab')

    for prop in props:
        for k in frag2score[prop]:
            frag2score[prop][k] /= frag2cnt[k]  # mean property value of the fragment
    
        df = pd.DataFrame({'frag': frag2score[prop].keys(),
                           'score': frag2score[prop].values()})
        df['size'] = df['frag'].apply(lambda frag: dm.to_mol(frag).GetNumAtoms())
        df = df.sort_values(by='score', ascending=False)
        df.to_csv(f'vocab/{prop}.csv', index=False)
