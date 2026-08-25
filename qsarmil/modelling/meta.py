from __future__ import annotations

import os
import pickle
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import psutil
from qsarcons.consensus import GeneticSearch
from rdkit import RDLogger

from qsarmil.modelling.lazy import LazyMIL, _validate_accelerator
from qsarmil.utils.logging import print_step_header

RDLogger.DisableLog("rdApp.*")


class MultiConformerEstimator:
    """Shared LazyMIL + consensus-search pipeline behind MultiConformerRegressor/MultiConformerClassifier."""

    _task: str | None = None
    _val_size: float = 0.2

    def __init__(
        self,
        num_conf: int = 10,
        hopt: bool = False,
        num_cpu: int = 20,
        output_folder: str | None = None,
        verbose: bool = True,
        seed: int = 42,
        accelerator: str = "cpu",
    ) -> None:
        """Store the settings passed through to the underlying LazyMIL run.

        Args:
            num_conf (int): Number of conformers to generate per molecule.
            hopt (bool): Whether to hyperparameter-tune each estimator.
            num_cpu (int): Number of CPU threads to use for conformer generation.
            output_folder (str, optional): Directory for LazyMIL's CSVs; a fresh temp dir is created if omitted.
            verbose (bool): Whether to print progress from the underlying steps.
            seed (int): Random seed for the train/val split and everything LazyMIL seeds internally.
            accelerator (str): ``"cpu"`` or ``"gpu"`` - an explicit choice, never auto-detected. Used for
                training; :meth:`predict` and :meth:`load` can override it later.
        """
        super().__init__()

        self.num_conf = num_conf
        self.num_cpu = num_cpu
        self.hopt = hopt
        self.output_folder: str = output_folder or tempfile.mkdtemp(prefix="qsarmil_")
        self.verbose = verbose
        self.seed = seed
        self.accelerator = _validate_accelerator(accelerator)
        self.best_consensus: list[str] = []
        self._consensus_search: Any | None = None
        self._lazy_model: LazyMIL | None = None

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

        lazy_ml = LazyMIL(
            num_conf=self.num_conf,
            hopt=self.hopt,
            num_cpu=self.num_cpu,
            output_folder=self.output_folder,
            verbose=self.verbose,
            seed=self.seed,
            val_size=self._val_size,
            task=self._task,
            accelerator=self.accelerator,
        )
        lazy_ml.run(smiles, y)

        res_val = pd.read_csv(f"{self.output_folder}/val.csv")
        x_val, true_val = res_val.iloc[:, 2:], res_val.iloc[:, 1]

        if self.verbose:
            print_step_header(5, "Genetic model consensus search")

        start = time.time()
        cons_search = GeneticSearch(cons_size="auto", n_iter=50)
        best_cons = cons_search.run(x_val, true_val)
        elapsed_min = (time.time() - start) / 60
        mem_gb = psutil.Process().memory_info().rss / (1024**3)

        self.best_consensus = list(best_cons)
        self._consensus_search = cons_search
        self._lazy_model = lazy_ml

        if self.verbose:
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

        if self._lazy_model is not None and self._lazy_model.is_trained:
            res_test = self._lazy_model.predict(smiles, save=save, accelerator=accelerator)
        else:
            raise RuntimeError("LazyMIL model is not trained. Call `train` or `load` first.")

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

    def save(self, model_path: str | Path | None = None) -> None:
        """Serialize the trained model state to disk."""

        if not self.is_trained:
            raise RuntimeError("Model is not trained. Nothing to serialize.")

        model_path = Path(model_path or Path(self.output_folder) / "model.pkl")
        state = {
            "num_conf": self.num_conf,
            "hopt": self.hopt,
            "num_cpu": self.num_cpu,
            "verbose": self.verbose,
            "seed": self.seed,
            "accelerator": self.accelerator,
            "best_consensus": self.best_consensus,
            "consensus_search": self._consensus_search,
            "lazy_model": self._lazy_model,
        }

        # Some environments may not allow pickling the qsarcons search object.
        try:
            pickle.dumps(state["consensus_search"])
        except (pickle.PickleError, AttributeError, TypeError):
            state["consensus_search"] = None

        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(
        cls, model_path: str | Path, output_folder: str | None = None, accelerator: str | None = None
    ) -> MultiConformerEstimator:
        """Load a serialized model state.

        Args:
            model_path (str | Path): Path to a file written by :meth:`save`.
            output_folder (str, optional): Directory for LazyMIL's CSVs; a fresh temp dir is created if omitted.
            accelerator (str, optional): ``"cpu"`` or ``"gpu"`` to override the accelerator this model was
                trained with for all future :meth:`predict` calls - e.g. load a GPU-trained model but run
                inference on CPU from now on. Defaults to whatever accelerator was used at training time.
        """

        model_path = Path(model_path)
        with model_path.open("rb") as f:
            state = pickle.load(f)

        model = cls(
            num_conf=state["num_conf"],
            hopt=state["hopt"],
            num_cpu=state["num_cpu"],
            output_folder=output_folder,
            verbose=state["verbose"],
            seed=state["seed"],
            accelerator=accelerator if accelerator is not None else state.get("accelerator", "cpu"),
        )
        model.best_consensus = list(state["best_consensus"])
        model._consensus_search = state.get("consensus_search")
        model._lazy_model = state.get("lazy_model")
        if model._lazy_model is not None:
            model._lazy_model.output_folder = model.output_folder
            model._lazy_model.accelerator = model.accelerator
            if os.path.exists(model.output_folder):
                shutil.rmtree(model.output_folder)
            os.makedirs(model.output_folder)
        return model


class MultiConformerRegressor(MultiConformerEstimator):
    """MultiConformerEstimator pipeline for continuous (regression) targets."""

    _task = "continuous"


class MultiConformerClassifier(MultiConformerEstimator):
    """MultiConformerEstimator pipeline for binary classification targets."""

    _task = "binary"

