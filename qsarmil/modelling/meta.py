from __future__ import annotations

import datetime
import os
from collections.abc import Sequence
from typing import Any

from qsarcons.consensus import GeneticSearch
from rdkit import RDLogger
from sklearn.model_selection import train_test_split

from qsarmil.modelling.lazy import LazyMIL

RDLogger.DisableLog("rdApp.*")


class MultiConformerEstimator:
    """Shared LazyMIL + consensus-search pipeline behind MultiConformerRegressor/MultiConformerClassifier."""

    _task: str | None = None
    _val_size: float = 0.2

    def __init__(
        self,
        num_conf: int = 10,
        hopt: bool = False,
        num_cpu: int = os.cpu_count() or 1,
        output_folder: str | None = None,
        verbose: bool = True,
        random_seed: int = 42,
        accelerator: str = "cpu",
    ) -> None:
        """Build the underlying LazyMIL, defaulting output_folder to a timestamped folder name if not given."""
        super().__init__()

        self.random_seed = random_seed
        self.verbose = verbose
        output_folder = output_folder or datetime.datetime.now().strftime("qsarmil_%d_%m_%Y_%H_%M_%S")  # noqa: DTZ005
        self._lazy_model = LazyMIL(
            task=self._task,
            num_conf=num_conf,
            hopt=hopt,
            num_cpu=num_cpu,
            output_folder=output_folder,
            verbose=verbose,
            random_seed=random_seed,
            accelerator=accelerator,
        )
        self.best_consensus: list[str] = []
        self._consensus_search: Any | None = None

    @property
    def output_folder(self) -> str:
        """Directory holding this model's files (train.csv/val.csv/test.csv)."""

        return self._lazy_model.output_folder

    def train_predict(self, smiles_train: Sequence[str], y_train: Sequence[Any], smiles_test: Sequence[str]) -> list[Any]:
        """Train, select a genetic model consensus, and predict on new SMILES - all in one call."""

        smi_train_all, y_train_all = list(smiles_train), list(y_train)
        idx_train, idx_val = train_test_split(
            range(len(smi_train_all)), test_size=self._val_size, random_state=self.random_seed
        )
        smi_train = [smi_train_all[i] for i in idx_train]
        y_train_split = [y_train_all[i] for i in idx_train]
        smi_val = [smi_train_all[i] for i in idx_val]
        y_val = [y_train_all[i] for i in idx_val]

        _, result_df_val, result_df_test = self._lazy_model.run(
            smi_train, y_train_split, smi_val, y_val, list(smiles_test)
        )

        x_val, true_val = result_df_val.iloc[:, 2:], result_df_val.iloc[:, 1]

        if self.verbose:
            print("Step-4. Genetic consensus search")

        cons_search = GeneticSearch(cons_size="auto", n_iter=50)
        best_cons = cons_search.run(x_val, true_val)

        self.best_consensus = list(best_cons)
        self._consensus_search = cons_search

        if self.verbose:
            print("Best genetic consensus:")
            for name in self.best_consensus:
                print(f"  -{name}")

        x_test = result_df_test.iloc[:, 1:]
        missing_cols = [c for c in self.best_consensus if c not in x_test.columns]
        if missing_cols:
            raise ValueError("Consensus references missing model columns: " + ", ".join(missing_cols))

        pred_test = list(self._consensus_search.predict(x_test[self.best_consensus]))
        result_df_test.to_csv(os.path.join(self.output_folder, "test.csv"), index=False)

        return pred_test


class MultiConformerRegressor(MultiConformerEstimator):
    """MultiConformerEstimator pipeline for continuous (regression) targets."""

    _task = "continuous"


class MultiConformerClassifier(MultiConformerEstimator):
    """MultiConformerEstimator pipeline for binary classification targets."""

    _task = "binary"
