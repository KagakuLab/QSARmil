from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from rdkit.Chem import Mol

from qsarmil.utils.logging import FailedDescriptor

_EXTREME_VALUE_THRESHOLD = 1e25

class DescriptorWrapper:
    """Compute molecular descriptors for a bag of conformers, for both RDKit-based and external transformers."""

    def __init__(self, transformer: Callable[..., np.ndarray], verbose: bool = True) -> None:
        """Store the descriptor transformer and verbosity setting.

        Args:
            transformer (callable): Descriptor function or object.
            verbose (bool): Whether to show progress bar.
        """
        super().__init__()
        self.transformer = transformer
        self.verbose = verbose
        self._keep_mask: np.ndarray | None = None

    def __call__(self, mols: list[Mol], *args: Any, **kwargs: Any) -> np.ndarray | FailedDescriptor:
        """Compute the raw descriptor bag (one vector per conformer) for a single molecule's bag of conformers."""
        return self._transform(mols)

    def _confs_to_descriptors(self, mols: list[Mol]) -> np.ndarray:
        """Convert a bag of single-conformer molecules into a descriptor matrix."""

        bag = [self.transformer(mol, conformer_id=0).flatten() for mol in mols]
        return np.array(bag)

    def _transform(self, mols: list[Mol]) -> np.ndarray | FailedDescriptor:
        """Compute descriptors for a single molecule's bag of conformers."""
        try:
            x = self._confs_to_descriptors(mols)
        except Exception as e:
            print(e)
            x = FailedDescriptor(mols)
        return x

    def run(self, list_of_confs: Sequence[list[Mol]]) -> list[np.ndarray]:
        """Compute descriptors for a list of bags, dropping any column that's NaN/extreme for at least one
        conformer (in any molecule seen so far by this instance) - reported by row and reused automatically
        on every later call to this same instance, so e.g. train/val/test all end up with the same columns
        without needing to pass anything extra around.

        Raises:
            ValueError: If any bag is a :class:`~qsarmil.utils.logging.FailedDescriptor` (descriptor
                calculation failed for that molecule) - listing exactly which ones and why, rather than
                letting it surface later as an opaque error somewhere downstream.
        """

        total = len(list_of_confs)

        # 1. Calculate descriptors
        orig_bags = []
        for i, mols in enumerate(list_of_confs, 1):
            orig_bags.append(self._transform(mols))
            if self.verbose:
                print(f"Calculating descriptors: {i}/{total}", end="\r", flush=True)

        stacked = np.vstack(orig_bags).astype(float)
        stacked[np.abs(stacked) >= _EXTREME_VALUE_THRESHOLD] = np.nan

        if self._keep_mask is None:
            bad_mask = np.isnan(stacked).any(axis=0)
            self._keep_mask = ~bad_mask

        cleaned_bags = []
        for bag in orig_bags:
            bag = np.array(bag, dtype=float)
            bag[np.abs(bag) >= _EXTREME_VALUE_THRESHOLD] = np.nan
            cleaned_bags.append(bag[:, self._keep_mask])

        return cleaned_bags


