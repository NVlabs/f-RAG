# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from wenhao-gao/mol_opt.
#
# Source:
# https://github.com/wenhao-gao/mol_opt/blob/main/main/graph_ga/run.py
#
# The license for this can be found in license_thirdparty/LICENSE_PMO.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import numpy as np
import datamol as dm
import ga.crossover as co
import ga.mutate as mu


def choose_parents(population_mol, population_scores):
    population_scores = [s + 1e-10 for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    parents = np.random.choice(population_mol, p=population_probs, size=2)
    return parents


def reproduce(population, mutation_rate):
    population_mol = [dm.to_mol(smiles) for prop, smiles in population]
    population_scores = [prop for prop, smiles in population]

    for _ in range(1000):
        parent_a, parent_b = choose_parents(population_mol, population_scores)
        new_child = co.crossover(parent_a, parent_b)
        if new_child is not None:
            new_child = mu.mutate(new_child, mutation_rate)
        if new_child is not None:
            return dm.to_smiles(new_child)
