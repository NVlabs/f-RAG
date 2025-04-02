# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from datamol-io/safe.
#
# Source:
# https://github.com/datamol-io/safe/blob/main/safe/utils.py
#
# The license for this can be found in license_thirdparty/LICENSE_SAFE.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import itertools
import datamol as dm
import numpy as np
from rdkit import Chem


class MolSlicer:
    BOND_SPLITTERS = [
        # two atoms connected by a non ring single bond, one of each is not in a ring and at least two heavy neighbor
        "[R:1]-&!@[!R;!D1:2]",
        # two atoms in different rings linked by a non-ring single bond
        "[R:1]-&!@[R:2]",
    ]
    _BOND_BUFFER = 1  # buffer around substructure match size.
    MAX_CUTS = 2  # maximum number of cuts. Here we need two cuts for head-linker-tail.

    def __init__(self, shortest_linker=False, min_linker_size=0, require_ring_system=True):
        self.bond_splitters = [Chem.MolFromSmarts(x) for x in self.BOND_SPLITTERS]
        self.shortest_linker = shortest_linker
        self.min_linker_size = min_linker_size
        self.require_ring_system = require_ring_system

    def get_ring_system(self, mol):
        """Get the list of ring system from a molecule"""
        mol.UpdatePropertyCache()
        ri = mol.GetRingInfo()
        systems = []
        for ring in ri.AtomRings():
            ring_atoms = set(ring)
            cur_system = []  # keep a track of ring system
            for system in systems:
                if len(ring_atoms.intersection(system)) > 0:
                    ring_atoms = ring_atoms.union(system)  # merge ring system that overlap
                else:
                    cur_system.append(system)
            cur_system.append(ring_atoms)
            systems = cur_system
        return systems

    def _bond_selection_from_max_cuts(self, bond_list, dist_mat):
        """Select bonds based on maximum number of cuts allowed"""
        # for now we are just implementing to 2 max cuts algorithms
        if self.MAX_CUTS != 2:
            raise ValueError(f"Only MAX_CUTS=2 is supported, got {self.MAX_CUTS}")

        bond_pdist = np.full((len(bond_list), len(bond_list)), -1)
        for i in range(len(bond_list)):
            for j in range(i, len(bond_list)):
                # we get the minimum topological distance between bond to cut
                bond_pdist[i, j] = bond_pdist[j, i] = min(
                    [dist_mat[a1, a2] for a1, a2 in itertools.product(bond_list[i], bond_list[j])]
                )

        masked_bond_pdist = np.ma.masked_less_equal(bond_pdist, self.min_linker_size)

        if self.shortest_linker:
            return np.unravel_index(np.ma.argmin(masked_bond_pdist), bond_pdist.shape)
        return np.unravel_index(np.ma.argmax(masked_bond_pdist), bond_pdist.shape)

    def _get_bonds_to_cut(self, mol):
        """Get possible bond to cuts"""
        # use this if you want to enumerate yourself the possible cuts
        ring_systems = self.get_ring_system(mol)
        candidate_bonds = []
        ring_query = Chem.rdqueries.IsInRingQueryAtom()

        for query in self.bond_splitters:
            bonds = mol.GetSubstructMatches(query, uniquify=True)
            cur_unique_bonds = [set(cbond) for cbond in candidate_bonds]
            # do not accept bonds part of the same ring system or already known
            for b in bonds:
                bond_id = mol.GetBondBetweenAtoms(*b).GetIdx()
                bond_cut = Chem.GetMolFrags(
                    Chem.FragmentOnBonds(mol, [bond_id], addDummies=False), asMols=True
                )
                can_add = not self.require_ring_system or all(
                    len(frag.GetAtomsMatchingQuery(ring_query)) > 0 for frag in bond_cut
                )
                if can_add and not (
                    set(b) in cur_unique_bonds or any(x.issuperset(set(b)) for x in ring_systems)
                ):
                    candidate_bonds.append(b)
        return candidate_bonds

    def _fragment_mol(self, mol, bonds):
        """Fragment molecules on bonds and return head, linker, tail combination"""
        tmp = Chem.rdmolops.FragmentOnBonds(mol, [b.GetIdx() for b in bonds])
        _frags = list(Chem.GetMolFrags(tmp, asMols=True))
        # linker is the one with 2 dummy atoms
        linker_pos = 0
        for pos, _frag in enumerate(_frags):
            if sum([at.GetSymbol() == "*" for at in _frag.GetAtoms()]) == 2:
                linker_pos = pos
                break
        linker = _frags.pop(linker_pos)
        head, tail = _frags
        return (head, linker, tail)

    def __call__(self, mol):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        # remove salt and solution
        mol = dm.keep_largest_fragment(mol)
        Chem.rdDepictor.Compute2DCoords(mol)
        dist_mat = Chem.rdmolops.GetDistanceMatrix(mol)

        candidate_bonds = self._get_bonds_to_cut(mol)

        # we have all the candidate bonds we can cut
        # now we need to pick the most plausible bonds
        selected_bonds = [mol.GetBondBetweenAtoms(a1, a2) for (a1, a2) in candidate_bonds]

        # CASE 1: no bond to cut ==> only head
        if len(selected_bonds) == 0:
            return (mol, None, None)

        # CASE 2: only one bond ==> linker is empty
        if len(selected_bonds) == 1:
            # there is no linker
            tmp = Chem.rdmolops.FragmentOnBonds(mol, [b.GetIdx() for b in selected_bonds])
            head, tail = Chem.GetMolFrags(tmp, asMols=True)
            return (head, None, tail)

        # CASE 3: we select the most plausible bond to cut on ourselves
        choice = self._bond_selection_from_max_cuts(candidate_bonds, dist_mat)
        selected_bonds = [selected_bonds[c] for c in choice]
        return self._fragment_mol(mol, selected_bonds)
    

class MolSlicerForSAFEEncoder(MolSlicer):
    def __call__(self, mol):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        # remove salt and solution
        mol = dm.keep_largest_fragment(mol)
        Chem.rdDepictor.Compute2DCoords(mol)
        dist_mat = Chem.rdmolops.GetDistanceMatrix(mol)

        candidate_bonds = self._get_bonds_to_cut(mol)
        selected_bonds = [mol.GetBondBetweenAtoms(a1, a2) for (a1, a2) in candidate_bonds]
        assert len(selected_bonds) != 0     # only head cases

        # CASE 3: we select the most plausible bond to cut on ourselves
        if len(selected_bonds) >= 2:
            choice = self._bond_selection_from_max_cuts(candidate_bonds, dist_mat)
            selected_bonds = [selected_bonds[c] for c in choice]
        
        for bond in selected_bonds:
            yield (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    