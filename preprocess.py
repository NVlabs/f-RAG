# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for f-RAG. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import re
import json
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
from datasets import load_dataset
import safe as sf
from rdkit import RDLogger, DataStructs, Chem
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
from fusion.slicer import MolSlicerForSAFEEncoder


def canonicalize(smiles):
    smiles = re.sub(r'\[\*:\d+\]', '*', smiles)
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def prepare_attach(smiles):
    smiles = re.sub(r'\[\*:\d+\]', '*', smiles)
    return re.sub(r'\*', '[1*]', smiles)


def attach(frag1, frag2, idx=0):
    rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
    mols = rxn.RunReactants((Chem.MolFromSmiles(frag1), Chem.MolFromSmiles(frag2)))
    return Chem.MolToSmiles(mols[idx][0])


def get_fps(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)


if __name__ == '__main__':
    if not os.path.exists('data/zinc250k_train.csv'):
        num_retrieve = 10

        df = pd.read_csv('data/zinc250k.csv')
        slicer = MolSlicerForSAFEEncoder(shortest_linker=True)

        arm_idx_list, arm_safe_list, arm_list = [], [], []
        linker_idx_list, linker_safe_list, linker_list = [], [], []
        for i, smiles in enumerate(tqdm(df['smiles'], desc='Decomposing')):
            try:
                safestr = sf.encode(Chem.MolFromSmiles(df['smiles'].iloc[i]), slicer=slicer)
                frags = safestr.split('.')
                for j, frag in enumerate(frags):
                    # permutate to place the frag at last
                    perm_safe = '.'.join(frags[:j] + frags[j + 1:] + [frags[j]])
                    frag = canonicalize(sf.decode(frag, remove_dummies=False))
                    if frag.count('*') == 1:
                        arm_idx_list.append(i)
                        arm_safe_list.append(perm_safe)
                        arm_list.append(frag)
                    else:
                        linker_idx_list.append(i)
                        linker_safe_list.append(perm_safe)
                        linker_list.append(frag)
            except KeyboardInterrupt:
                quit()
            except:
                continue

        df_arm = pd.DataFrame({'idx': arm_idx_list,
                            'safe': arm_safe_list,
                            'frag': arm_list})
        df_linker = pd.DataFrame({'idx': linker_idx_list,
                                'safe': linker_safe_list,
                                'frag': linker_list})
        
        df_arm['fps'] = df_arm['frag'].apply(get_fps)
        df_linker['fps'] = df_linker['frag'].apply(get_fps)
        
        with open('data/zinc250k_test_idx.json') as f:
            test_idx = set(json.load(f))
        train_idx = {i for i in range(len(df)) if i not in test_idx}
        
        for df, frag_type in zip([df_arm, df_linker], ['arm', 'linker']):
            idx_list, input_list, retrieved_list = [], [], []
            for i in trange(len(df), desc=f'Calculating similarity for {frag_type}'):
                df_tmp = df.copy()
                df_tmp = df_tmp.drop_duplicates(subset='frag')
                df_tmp['sim'] = DataStructs.BulkTanimotoSimilarity(df['fps'].iloc[i],
                                                                    np.array(df_tmp['fps']))
                df_tmp = df_tmp[df_tmp['sim'] < 1]  # remove identical fragments
                df_tmp = df_tmp.sort_values(by='sim', ascending=False)
                retrieved = df_tmp['frag'].iloc[:num_retrieve].tolist()
                
                try:
                    # new smiles where the fragment is replaced with the most similar fragment
                    most_similar_frag = prepare_attach(retrieved[0])
                    retrieved_frags = '.'.join([df['frag'].iloc[i]] + retrieved[1:])
                    
                    frags = df['safe'].iloc[i].split('.')
                    rest, frag = frags[:-1], frags[-1]
                    if frag_type == 'arm':
                        rest = sf.decode('.'.join(rest), remove_dummies=False)
                        if rest is None:
                            continue
                        rest = prepare_attach(rest)
                        attached_smiles = attach(rest, most_similar_frag)
                    if frag_type == 'linker':
                        assert len(rest) == 2
                        rest1, rest2 = sf.decode(rest[0], remove_dummies=False), sf.decode(rest[1], remove_dummies=False)
                        if rest1 is None or rest2 is None:
                            continue
                        rest1, rest2 = prepare_attach(rest1), prepare_attach(rest2)
                        attached_smiles = attach(attach(rest1, most_similar_frag), rest2)

                    new_safe = sf.encode(Chem.MolFromSmiles(attached_smiles), slicer=slicer)
                    frags = new_safe.split('.')
                    for j, frag in enumerate(frags):
                        if canonicalize(sf.decode(frag, remove_dummies=False)) == retrieved[0]:
                            break
                    else:
                        continue
                    
                    # permutate to place the most_similar_frag at last
                    new_safe = '.'.join(frags[:j] + frags[j + 1:] + [frags[j]])

                except KeyboardInterrupt:
                    quit()
                except:
                    continue

                idx_list.append(df['idx'].iloc[i])
                input_list.append(new_safe)
                retrieved_list.append(retrieved_frags)

            df = pd.DataFrame({'idx': idx_list,
                            'input': input_list,
                            'retrieved': retrieved_list})
            df = df[[i in train_idx for i in df['idx']]]
            del df['idx']
            df.to_csv(f'data/zinc250k_{frag_type}.csv', index=False)
        
        df_arm = pd.read_csv('data/zinc250k_arm.csv')
        df_linker = pd.read_csv('data/zinc250k_linker.csv')
        df = pd.concat([df_arm, df_linker])
        df.to_csv('data/zinc250k_train.csv', index=False)
        print(f'{len(df)} training samples')
        os.remove('data/zinc250k_arm.csv')
        os.remove('data/zinc250k_linker.csv')
    
    dataset = load_dataset('csv', data_files={'train': 'data/zinc250k_train.csv'})
    dataset.save_to_disk('data/zinc250k')
