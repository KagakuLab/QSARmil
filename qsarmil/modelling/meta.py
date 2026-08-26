from __future__ import annotations

import os
import time
from typing import Any, Sequence

import pandas as pd
import psutil
from qsarcons.consensus import GeneticSearch
from rdkit import RDLogger

from qsarmil.modelling.lazy import LazyMIL
from qsarmil.utils.logging import print_step_header

RDLogger.DisableLog("rdApp.*")


class MultiConformerEstimator:
    """Shared LazyMIL + consensus-search pipeline behind MultiConformerRegressor/MultiConformerClassifier.

    All the training/inference settings (``num_conf``, ``hopt``, ``accelerator``, etc.) live on the
    ``LazyMIL`` this builds in :meth:`__init__` - this class itself only owns the consensus-search result.
    Train and predict within the same process/session - there's no serialization; use :meth:`train` then
    :meth:`predict` in one run.
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
            accelerator (str): ``"cpu"`` or ``"gpu"`` - an explicit choice, never auto-detected.
        """
        super().__init__()

        self._lazy_model = LazyMIL(
            num_conf=num_conf,
            hopt=hopt,
            num_cpu=num_cpu,
            output_folder=output_folder,
            verbose=verbose,
            seed=seed,
            val_size=self._val_size,
            task=self._task,
            accelerator=accelerator,
        )
        self.best_consensus: list[str] = []
        self._consensus_search: Any | None = None

    @property
    def output_folder(self) -> str:
        """Directory holding this model's intermediate files (``train.csv``/``val.csv``/``test.csv``)."""

        return self._lazy_model.output_folder

    @property
    def is_trained(self) -> bool:
        """Whether :meth:`train` has produced a reusable consensus."""

        return bool(self.best_consensus)

    def train(self, smiles: Sequence[str], y: Sequence[Any]) -> MultiConformerEstimator:
        """Train/model-select once and cache everything required for later prediction.

        Args:
            smiles (Sequence[str]): Training SMILES strings.
            y (Sequence[Any]): Target property value for each SMILES, same length/order as ``smiles``.

        Returns:
            MultiConformerEstimator: This instance, now containing trained state.
        """

        self._lazy_model.run(smiles, y)

        res_val = pd.read_csv(f"{self.output_folder}/val.csv")
        x_val, true_val = res_val.iloc[:, 2:], res_val.iloc[:, 1]

        if self._lazy_model.verbose:
            print_step_header(5, "Genetic model consensus search")

        start = time.time()
        cons_search = GeneticSearch(cons_size="auto", n_iter=50)
        best_cons = cons_search.run(x_val, true_val)
        elapsed_min = (time.time() - start) / 60
        mem_gb = psutil.Process().memory_info().rss / (1024**3)

        self.best_consensus = list(best_cons)
        self._consensus_search = cons_search

        if self._lazy_model.verbose:
            print(f"> Finished in {elapsed_min:.2f} min | Memory usage: {mem_gb:.3f} G")
            print("> Best genetic consensus:")
            for name in self.best_consensus:
                print(f"       -{name}")

        return self

    def predict(self, smiles: Sequence[str], save: bool = False) -> list[Any]:
        """Predict for new SMILES using the stored trained state.

        Args:
            smiles (Sequence[str]): SMILES strings to predict on.
            save (bool): Whether to also write LazyMIL's per-model predictions to ``test.csv``.

        Returns:
            list: Predicted property value for each input SMILES, same order as ``smiles``.
        """

        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call `train` first.")

        res_test = self._lazy_model.predict(smiles, save=save)
        x_test = res_test.iloc[:, 1:]

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
