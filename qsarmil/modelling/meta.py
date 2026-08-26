from __future__ import annotations

import os
import pickle
import time
from typing import Any, Sequence

import pandas as pd
import psutil
from qsarcons.consensus import GeneticSearch
from rdkit import RDLogger

from qsarmil.modelling.lazy import LazyMIL, _validate_accelerator
from qsarmil.utils.logging import print_step_header

RDLogger.DisableLog("rdApp.*")

_ESTIMATOR_FILENAME = "estimator.pkl"


class MultiConformerEstimator:
    """Shared LazyMIL + consensus-search pipeline behind MultiConformerRegressor/MultiConformerClassifier.

    All the training/inference settings (``num_conf``, ``hopt``, ``accelerator``, etc.) live on the
    ``LazyMIL`` this builds in :meth:`__init__` - this class itself only owns the consensus-search result.
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
            accelerator (str): ``"cpu"`` or ``"gpu"`` - an explicit choice, never auto-detected. Used for
                training; :meth:`predict` and :meth:`load` can override it later.
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
        """Directory holding this model's files - the only thing :meth:`save`/:meth:`load` need to know."""

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

    def predict(self, smiles: Sequence[str], save: bool = False, accelerator: str | None = None) -> list[Any]:
        """Predict for new SMILES using the stored trained state.

        Args:
            smiles (Sequence[str]): SMILES strings to predict on.
            save (bool): Whether to also write LazyMIL's per-model predictions to ``test.csv``.
            accelerator (str, optional): ``"cpu"`` or ``"gpu"`` to run inference on, overriding
                :attr:`accelerator` for this call only - e.g. predict on CPU for a model trained on GPU.

        Returns:
            list: Predicted property value for each input SMILES, same order as ``smiles``.
        """

        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call `train` or `load` first.")

        res_test = self._lazy_model.predict(smiles, save=save, accelerator=accelerator)
        x_test = res_test.iloc[:, 1:]

        missing_cols = [c for c in self.best_consensus if c not in x_test.columns]
        if missing_cols:
            raise ValueError(
                "Serialized consensus references missing model columns: "
                + ", ".join(missing_cols)
            )

        # Prefer the original qsarcons predictor when it can be serialized,
        # but keep a robust fallback for environments where that object is
        # not pickle-friendly.
        if self._consensus_search is not None and hasattr(self._consensus_search, "predict"):
            pred_test = self._consensus_search.predict(x_test[self.best_consensus])
        else:
            pred_test = x_test[self.best_consensus].mean(axis=1).to_numpy()

        return list(pred_test)

    def __getstate__(self) -> dict[str, Any]:
        """Drop ``_consensus_search`` if it can't be pickled, warning instead of failing silently."""

        state = self.__dict__.copy()
        try:
            pickle.dumps(state["_consensus_search"])
        except (pickle.PickleError, AttributeError, TypeError):
            print(
                "Warning: the consensus search object could not be serialized and will be dropped. "
                "The loaded model will predict using the mean of the consensus models' predictions instead."
            )
            state["_consensus_search"] = None
        return state

    def save(self) -> None:
        """Serialize the trained model to ``{self.output_folder}/estimator.pkl``."""

        if not self.is_trained:
            raise RuntimeError("Model is not trained. Nothing to serialize.")

        path = os.path.join(self.output_folder, _ESTIMATOR_FILENAME)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, output_folder: str, accelerator: str | None = None) -> MultiConformerEstimator:
        """Load a model previously saved with :meth:`save`.

        Args:
            output_folder (str): The folder passed as ``output_folder`` at training time (or wherever it
                was moved to since) - the only thing you ever need to point at; internal file names/layout
                inside it are an implementation detail.
            accelerator (str, optional): ``"cpu"`` or ``"gpu"`` to override the accelerator this model was
                trained with for all future :meth:`predict` calls - e.g. load a GPU-trained model but run
                inference on CPU from now on. Defaults to whatever accelerator was used at training time.

        Returns:
            MultiConformerEstimator: The exact object (and subclass) that was saved.
        """

        estimator_path = os.path.join(output_folder, _ESTIMATOR_FILENAME)
        if not os.path.exists(estimator_path):
            raise FileNotFoundError(
                f"No {_ESTIMATOR_FILENAME} found in {output_folder!r} - is this a QSARmil model folder?"
            )

        with open(estimator_path, "rb") as f:
            model: MultiConformerEstimator = pickle.load(f)

        model._lazy_model.output_folder = output_folder
        if accelerator is not None:
            model._lazy_model.accelerator = _validate_accelerator(accelerator)

        if not os.path.exists(model._lazy_model.models_path()):
            raise FileNotFoundError(
                f"No models.pkl found in {output_folder!r} - the model folder may be incomplete."
            )

        return model


class MultiConformerRegressor(MultiConformerEstimator):
    """MultiConformerEstimator pipeline for continuous (regression) targets."""

    _task = "continuous"


class MultiConformerClassifier(MultiConformerEstimator):
    """MultiConformerEstimator pipeline for binary classification targets."""

    _task = "binary"
