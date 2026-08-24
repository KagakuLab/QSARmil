from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from rdkit.Chem import Mol

from qsarmil.utils.logging import FailedDescriptor

# Absolute descriptor values beyond this are treated as broken/unreliable,
# the same as a NaN - some 3D descriptor calculators occasionally blow up to
# numerically meaningless magnitudes on degenerate geometries.
_EXTREME_VALUE_THRESHOLD = 1e25


class DescriptorWrapper:
    """Wrapper to compute molecular descriptors for multiple conformers in
    parallel.

    Converts a molecule's bag of single-conformer ``Mol`` objects into a
    bag of descriptor vectors, one per conformer, with optional
    parallelization and progress tracking. Works the same regardless of
    whether ``transformer`` is an RDKit-based descriptor
    (:class:`~qsarmil.descriptor.rdkit.RDKitDescriptor3D` and subclasses)
    or an external one (e.g. a MolFeat calculator) - see :meth:`postprocess`.

    Args:
        transformer (callable): Descriptor function or object that accepts a molecule
            and optional conformer ID, returning a descriptor vector.
        verbose (bool): Whether to display a progress bar.
    """

    def __init__(self, transformer: Callable[..., np.ndarray], verbose: bool = True) -> None:
        """Initialize the descriptor wrapper.

        Args:
            transformer (callable): Descriptor function or object.
            verbose (bool): Whether to show progress bar.
        """
        super().__init__()
        self.transformer = transformer
        self.verbose = verbose

    def __call__(self, mols: list[Mol], *args: Any, **kwargs: Any) -> np.ndarray | FailedDescriptor:
        """Compute the descriptor bag for a single molecule.

        Args:
            mols (list[Mol]): A bag of single-conformer molecules to compute
                descriptors for (one row of output per conformer).

        Returns:
            np.ndarray: One descriptor vector per conformer in the bag.
        """
        return self._transform(mols)

    def _bag_to_descriptors(self, mols: list[Mol]) -> np.ndarray:
        """Convert a bag of single-conformer molecules into a descriptor matrix."""

        bag = [self.transformer(mol, conformer_id=0).flatten() for mol in mols]
        return np.array(bag)

    def _transform(self, mols: list[Mol]) -> np.ndarray | FailedDescriptor:
        """Compute descriptors for a single molecule's bag of conformers."""
        try:
            x = self._bag_to_descriptors(mols)
        except Exception as e:
            print(e)
            x = FailedDescriptor(mols)
        return x

    def run(self, list_of_bags: Sequence[list[Mol]]) -> list[np.ndarray | FailedDescriptor]:
        """Compute descriptors for a list of molecules' conformer bags."""

        total = len(list_of_bags)
        results = []
        for i, mols in enumerate(list_of_bags, 1):
            results.append(self._transform(mols))
            if self.verbose:
                print(f"Calculating descriptors: {i}/{total}", end="\r", flush=True)

        if self.verbose:
            print(f"Calculating descriptors: {total}/{total}")

        return results

    def postprocess(
        self,
        bags: list[np.ndarray],
        verbose: bool = False,
        col_stats: dict[str, np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        """Drop unreliable descriptor columns from a list of per-molecule bags.

        Non-finite values (NaN, and values whose magnitude reaches
        :data:`_EXTREME_VALUE_THRESHOLD`) are treated as missing. Any column
        that's missing for even a single conformer, in any molecule, is
        dropped entirely - we don't impute a column mean here, since in
        practice a partially-missing 3D descriptor column has always meant
        the column isn't reliable for this dataset, not that it's worth
        salvaging.

        Works the same regardless of what produced ``bags`` (an RDKit
        descriptor, a MolFeat calculator, or anything else `run` was pointed
        at), since it operates purely on the resulting numeric matrix.

        Args:
            bags (list[np.ndarray]): Per-molecule descriptor matrices
                (conformers x raw features), one per molecule.
            verbose (bool): Whether to print which columns were dropped and
                why. Only takes effect while computing stats fresh
                (``col_stats=None``) - when reusing stats from a prior call,
                no new decision is being made, so nothing is printed.
            col_stats (dict, optional): Stats returned by an earlier call -
                pass the training split's stats here when cleaning
                validation/test/inference data, so every split ends up with
                the exact same columns instead of each one making its own
                (potentially different) decision.

        Returns:
            tuple[list[np.ndarray], dict]: ``(cleaned_bags, col_stats)``,
            where ``col_stats`` is ``{"keep_mask": np.ndarray}`` - reuse it
            via the ``col_stats`` argument to clean another split consistently.
        """

        stacked = np.vstack(bags).astype(float)
        stacked[np.abs(stacked) >= _EXTREME_VALUE_THRESHOLD] = np.nan

        if col_stats is None:
            bad_mask = np.isnan(stacked).any(axis=0)
            keep_mask = ~bad_mask

            if verbose and bad_mask.any():
                self._report_removed_columns(bad_mask, stacked)

            col_stats = {"keep_mask": keep_mask}
        else:
            keep_mask = col_stats["keep_mask"]

        cleaned_bags = []
        for bag in bags:
            bag = np.array(bag, dtype=float)
            bag[np.abs(bag) >= _EXTREME_VALUE_THRESHOLD] = np.nan
            cleaned_bags.append(bag[:, keep_mask])

        return cleaned_bags, col_stats

    def _report_removed_columns(self, bad_mask: np.ndarray, stacked: np.ndarray) -> None:
        """Print which descriptor columns were dropped, and why."""

        name = type(self.transformer).__name__
        columns = getattr(self.transformer, "columns", None)
        n_conformers = stacked.shape[0]

        print(
            f"Removed {int(bad_mask.sum())} of {len(bad_mask)} {name} descriptor column(s) "
            "(invalid for at least one conformer):"
        )
        for col_idx in np.where(bad_mask)[0]:
            n_bad = int(np.isnan(stacked[:, col_idx]).sum())
            label = columns[col_idx] if columns is not None else f"column {col_idx}"
            print(f"  - {label}: invalid for {n_bad}/{n_conformers} conformers")
