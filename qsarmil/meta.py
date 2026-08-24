from __future__ import annotations

import os
import pickle
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import psutil
from qsarcons.consensus import GeneticSearch
from rdkit import RDLogger

from qsarmil.lazy import LazyMIL
from qsarmil.utils.logging import print_step_header

RDLogger.DisableLog("rdApp.*")


class MultiConformerEstimator:
    """Shared implementation behind :class:`MultiConformerRegressor` and
    :class:`MultiConformerClassifier`.

    Wraps :class:`~qsarmil.lazy.LazyMIL` to train every built-in
    descriptor/estimator combination, then picks the best-performing
    consensus of models on the validation split via a genetic search
    (:class:`qsarcons.consensus.GeneticSearch`).

    Not meant to be instantiated directly - use one of the two subclasses
    so the task type is explicit rather than auto-detected from ``y``
    (auto-detection can misfire, e.g. a 2-value numeric target looks
    identical to a binary-classification target).
    """

    _task: str | None = None

    def __init__(
        self,
        num_conf: int = 10,
        hopt: bool = False,
        num_cpu: int = 20,
        output_folder: str | None = None,
        verbose: bool = True,
        seed: int = 42,
        val_size: float = 0.2,
    ) -> None:
        """Store the settings passed through to the underlying LazyMIL run.

        Args:
            num_conf (int): Number of conformers to generate per molecule.
            hopt (bool): Whether to hyperparameter-tune each estimator.
            num_cpu (int): Number of CPU threads to use for conformer generation.
            output_folder (str, optional): Directory for LazyMIL's
                intermediate prediction CSVs. If omitted, a fresh temporary
                directory is created.
            verbose (bool): Whether to print progress from the underlying steps.
            seed (int): Random seed used for the train/val split and for
                everything :class:`~qsarmil.lazy.LazyMIL` seeds internally
                (conformer embedding, molecule validation, hyperparameter
                search). Does **not** cover the final genetic consensus
                search - see the note in :meth:`train`.
            val_size (float): Fraction of the training data held out as a
                random validation split (used for consensus selection).
        """
        super().__init__()

        self.num_conf = num_conf
        self.num_cpu = num_cpu
        self.hopt = hopt
        self.output_folder: str = output_folder or tempfile.mkdtemp(prefix="qsarmil_")
        self.verbose = verbose
        self.seed = seed
        self.val_size = val_size
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
            y (Sequence[Any]): Target property value for each SMILES, same
                length and order as ``smiles``.

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
            val_size=self.val_size,
            task=self._task,
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

    def predict(self, smiles: Sequence[str], save: bool = False) -> list[Any]:
        """Predict for new SMILES using the stored trained state.

        Args:
            smiles (Sequence[str]): SMILES strings to predict on.
            save (bool): Whether to also write LazyMIL's per-model
                predictions to ``test.csv`` in ``output_folder``.

        Returns:
            list: Predicted property value for each input SMILES, in the
            same order as ``smiles``.
        """

        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call `train` or `load` first.")

        if self._lazy_model is not None and self._lazy_model.is_trained:
            res_test = self._lazy_model.predict(smiles, save=save)
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
            "val_size": self.val_size,
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
    def load(cls, model_path: str | Path, output_folder: str | None = None) -> MultiConformerEstimator:
        """Load a serialized model state."""

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
            val_size=state.get("val_size", 0.2),
        )
        model.best_consensus = list(state["best_consensus"])
        model._consensus_search = state.get("consensus_search")
        model._lazy_model = state.get("lazy_model")
        if model._lazy_model is not None:
            model._lazy_model.output_folder = model.output_folder
            if os.path.exists(model.output_folder):
                shutil.rmtree(model.output_folder)
            os.makedirs(model.output_folder)
        return model


def _check_continuous_target(y: Sequence[Any]) -> None:
    """Raise if ``y`` looks like classification labels rather than a
    continuous property.

    This can't be perfectly reliable (there's no way to tell "0.0 and 1.0
    happen to be the only two potency values in this dataset" from "0 and 1
    are class labels" without more context), so the check is deliberately
    narrow: it only flags targets whose dtype is boolean or integer, or
    that aren't numeric at all. A float target is always accepted, even
    with only a couple of distinct values - that's exactly the case
    :class:`MultiConformerRegressor` exists to support (see
    :class:`MultiConformerEstimator`'s docstring).

    Args:
        y (Sequence[Any]): Target values to check.

    Raises:
        ValueError: If ``y`` is non-numeric, boolean, or integer-dtyped.
    """

    y_arr = np.asarray(list(y))

    if y_arr.dtype == object or np.issubdtype(y_arr.dtype, np.str_):
        raise ValueError(
            "MultiConformerRegressor requires a continuous numeric target, but the values "
            f"provided are not numeric (dtype={y_arr.dtype}). If these are class labels, use "
            "MultiConformerClassifier instead."
        )

    if np.issubdtype(y_arr.dtype, np.bool_) or np.issubdtype(y_arr.dtype, np.integer):
        raise ValueError(
            "MultiConformerRegressor received a boolean/integer-only target, which looks like "
            "classification labels rather than a continuous property. If this is intentional "
            "(e.g. count data), convert y to float first; otherwise use MultiConformerClassifier."
        )


class MultiConformerRegressor(MultiConformerEstimator):
    """MultiConformerEstimator pipeline for continuous (regression) targets.

    See :class:`MultiConformerEstimator` for the full train/predict/save/load
    API. This subclass fixes the task to regression, so ``y`` is never
    passed through ``sklearn``'s auto-detection - a target with only two
    distinct numeric values (e.g. ``[1.0, 3.0]``) would otherwise be
    misdetected as binary classification.

    :meth:`train` additionally rejects targets that are clearly labels
    rather than a continuous property (booleans, integers, or non-numeric
    values) - see :func:`_check_continuous_target`.
    """

    _task = "continuous"

    def train(self, smiles: Sequence[str], y: Sequence[Any]) -> MultiConformerEstimator:
        """Train, after checking ``y`` doesn't look like classification labels.

        See :meth:`MultiConformerEstimator.train` for the full behavior.

        Raises:
            ValueError: If ``y`` looks like classification labels rather
                than a continuous property (see :func:`_check_continuous_target`).
        """

        _check_continuous_target(y)
        return super().train(smiles, y)


class MultiConformerClassifier(MultiConformerEstimator):
    """MultiConformerEstimator pipeline for binary classification targets.

    See :class:`MultiConformerEstimator` for the full train/predict/save/load
    API. This subclass fixes the task to binary classification.
    """

    _task = "binary"

