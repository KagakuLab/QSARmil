"""Shared fixtures for the qsarmil test suite."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem


def embed_mol(smiles: str, num_conf: int = 5, seed: int = 42) -> Chem.Mol:
    """Build a real, embedded, force-field-optimized RDKit molecule."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    for cid in conf_ids:
        AllChem.UFFOptimizeMolecule(mol, confId=cid)
    return mol


@pytest.fixture
def embedded_mol() -> Chem.Mol:
    """A real molecule (ibuprofen) with 5 embedded, optimized conformers."""
    return embed_mol("CC(C)Cc1ccc(cc1)C(C)C(=O)O", num_conf=5)


@pytest.fixture
def small_smiles_list() -> list[str]:
    """A handful of small, fast-to-embed, valid SMILES."""
    return ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]


class MockEstimator:
    """No-op MIL estimator: exercises real orchestration code without
    paying for real network training."""

    def __init__(self, supports_hopt: bool = True) -> None:
        self.mean_y: float = 0.0
        self.hopt_called = False
        if supports_hopt:
            # bound as an instance attribute so hasattr(obj, "hopt") only
            # holds for instances actually meant to support it
            self.hopt = self._hopt

    def _hopt(self, x: Any, y: Any, param_grid: dict[str, Any], verbose: bool = False) -> None:
        self.hopt_called = True
        self.last_param_grid = param_grid

    def fit(self, x: Any, y: Any) -> MockEstimator:
        self.mean_y = float(np.mean(list(y)))
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.full(len(x), self.mean_y)

    def __call__(self, accelerator: str | None = None) -> "MockEstimator":
        """Allow the mock instance to act as a factory; ignores accelerator like a non-milearn mock would."""
        self.accelerator = accelerator
        return self


@pytest.fixture
def mock_estimator() -> MockEstimator:
    return MockEstimator()


@pytest.fixture
def train_val_test_dfs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tiny continuous-target train/val/test dataframes, with one
    deliberately invalid SMILES mixed into train."""
    df_train = pd.DataFrame({0: ["CCO", "c1ccccc1", "not_a_valid_smiles!!!"], 1: [1.1, 2.2, 3.3]})
    df_val = pd.DataFrame({0: ["CCN", "CCC"], 1: [1.6, 2.6]})
    df_test = pd.DataFrame({0: ["CCCl", "CCF"], 1: [0.6, 1.6]})
    return df_train, df_val, df_test
