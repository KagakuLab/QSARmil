from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from qsarcons.consensus import GeneticSearch
from rdkit import RDLogger
from sklearn.model_selection import train_test_split

from qsarmil.modelling.lazy import LazyMIL

RDLogger.DisableLog("rdApp.*")


class MultiConformerEstimator:
    """Shared LazyMIL + consensus-search pipeline behind MultiConformerRegressor/MultiConformerClassifier.

    All the training/inference settings (``num_conf``, ``hopt``, etc.) live on the ``LazyMIL`` this
    builds in :meth:`__init__` - this class itself only owns the consensus-search result. There's no
    serialization and no separate predict() step: :meth:`train_predict` does everything in one call.
    """

    _task: str | None = None
    _val_size: float = 0.2

    def __init__(
        self,
        num_conf: int = 10,
        hopt: bool = False,
        num_cpu: int = os.cpu_count() or 1,
        output_folder: str | None = None,
        verbose: bool = True,
        seed: int = 42,
        accelerator: str = "cpu",
    ) -> None:
        """Build the underlying LazyMIL with these settings; see :class:`~qsarmil.modelling.lazy.LazyMIL`.

        Args:
            num_conf (int): Number of conformers to generate per molecule.
            hopt (bool): Whether to hyperparameter-tune each estimator.
            num_cpu (int): Number of CPU threads to use for conformer generation.
            output_folder (str, optional): Directory for the model's files; a fresh temp dir is created if omitted.
            verbose (bool): Whether to print progress from the underlying steps.
            seed (int): Random seed for the train/val split and everything LazyMIL seeds internally.
            accelerator (str): ``"cpu"`` or ``"gpu"`` - passed straight through to LazyMIL, which
                threads it into every estimator's construction and hyperparameter search.
        """
        super().__init__()

        self.seed = seed
        self.verbose = verbose
        self._lazy_model = LazyMIL(
            task=self._task,
            num_conf=num_conf,
            hopt=hopt,
            num_cpu=num_cpu,
            output_folder=output_folder,
            verbose=verbose,
            seed=seed,
            accelerator=accelerator,
        )
        self.best_consensus: list[str] = []
        self._consensus_search: Any | None = None

    @property
    def output_folder(self) -> str:
        """Directory holding this model's intermediate files (``train.csv``/``val.csv``/``test.csv``)."""

        return self._lazy_model.output_folder

    def train_predict(self, smiles_train: Sequence[str], y_train: Sequence[Any], smiles_test: Sequence[str]) -> list[Any]:
        """Train, select a genetic model consensus, and predict on new SMILES - all in one call.

        Args:
            smiles_train (Sequence[str]): Training SMILES strings.
            y_train (Sequence[Any]): Target property value for each SMILES, same length/order as
                ``smiles_train``. Internally split into train/validation.
            smiles_test (Sequence[str]): SMILES strings to predict on.

        Returns:
            list: Predicted property value for each input SMILES, same order as ``smiles_test``.
        """

        smi_train_all, y_train_all = list(smiles_train), list(y_train)
        idx_train, idx_val = train_test_split(
            range(len(smi_train_all)), test_size=self._val_size, random_state=self.seed
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
            print("Step-5. Genetic model consensus search")

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

        pred_test = self._consensus_search.predict(x_test[self.best_consensus])

        return list(pred_test)


class MultiConformerRegressor(MultiConformerEstimator):
    """MultiConformerEstimator pipeline for continuous (regression) targets."""

    _task = "continuous"


class MultiConformerClassifier(MultiConformerEstimator):
    """MultiConformerEstimator pipeline for binary classification targets."""

    _task = "binary"
