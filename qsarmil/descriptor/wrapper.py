from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from rdkit.Chem import Mol

from qsarmil.utils.logging import FailedMolecule

_EXTREME_VALUE_THRESHOLD = 1e25


class DescriptorWrapper:
    """Compute molecular descriptors for a bag of conformers, for both RDKit-based and external transformers."""

    def __init__(self, transformer: Callable[..., np.ndarray], verbose: bool = True) -> None:
        """Store the descriptor transformer and verbosity setting."""
        super().__init__()
        self.transformer = transformer
        self.verbose = verbose

    def __call__(self, mols: list[Mol], *args: Any, **kwargs: Any) -> np.ndarray | FailedMolecule:
        """Compute the raw descriptor bag (one vector per conformer) for a single molecule's bag of conformers."""
        return self._transform(mols)

    def _transform(self, mols: list[Mol]) -> np.ndarray | FailedMolecule:
        """Compute the descriptor matrix for one molecule's bag of conformers; failures become a sentinel."""
        try:
            bag = [self.transformer(mol, conformer_id=0).flatten() for mol in mols]
            return np.array(bag)
        except Exception as e:
            print(e)
            return FailedMolecule(mols, message="descriptor calculation failed")

    def run(self, list_of_confs: Sequence[list[Mol]]) -> list[np.ndarray | FailedMolecule]:
        """Compute descriptors for every bag, dropping columns that are NaN/extreme for at least one bag."""

        total = len(list_of_confs)

        raw_bags = []
        for i, mols in enumerate(list_of_confs, 1):
            raw_bags.append(self._transform(mols))
            if self.verbose:
                print(f"Calculating descriptors: {i}/{total}", end="\r", flush=True)

        valid_bags = [bag for bag in raw_bags if isinstance(bag, np.ndarray)]
        if not valid_bags:
            return raw_bags

        stacked = np.vstack(valid_bags).astype(float)
        stacked[np.abs(stacked) >= _EXTREME_VALUE_THRESHOLD] = np.nan

        keep_mask = ~np.isnan(stacked).any(axis=0)
        if not keep_mask.all():
            print(f"Removed {int((~keep_mask).sum())} of {len(keep_mask)} descriptor column(s) with extreme/invalid values.")

        cleaned_bags: list[np.ndarray | FailedMolecule] = []
        for bag in raw_bags:
            if not isinstance(bag, np.ndarray):
                cleaned_bags.append(bag)
                continue
            bag = bag.astype(float)
            bag[np.abs(bag) >= _EXTREME_VALUE_THRESHOLD] = np.nan
            cleaned_bags.append(bag[:, keep_mask])

        return cleaned_bags
