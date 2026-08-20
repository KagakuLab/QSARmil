from __future__ import annotations

import os
import pickle
import shutil
import tempfile
import json
from pathlib import Path
from typing import Any

import pandas as pd
from qsarcons.consensus import GeneticSearch
from rdkit import RDLogger
from sklearn.model_selection import train_test_split

from qsarmil.lazy import LazyMIL

RDLogger.DisableLog("rdApp.*")


class MultiConformerModel:
    """Lazy MIL pipeline with train/predict split and consensus model selection.

    Wraps :class:`~qsarmil.lazy.LazyMIL` to train every built-in
    descriptor/estimator combination, then picks the best-performing
    consensus of models on the validation split via a genetic search
    (:class:`qsarcons.consensus.GeneticSearch`) and applies it to the test set.
    """

    def __init__(
        self,
        num_conf: int = 10,
        hopt: bool = False,
        num_cpu: int = 20,
        output_folder: str | None = None,
        verbose: bool = True,
        seed: int = 42,
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
        """
        super().__init__()

        self.num_conf = num_conf
        self.num_cpu = num_cpu
        self.hopt = hopt
        self.output_folder: str = output_folder or tempfile.mkdtemp(prefix="qsarmil_")
        self.verbose = verbose
        self.seed = seed
        self.best_consensus: list[str] = []
        self._consensus_search: Any | None = None
        self._lazy_model: LazyMIL | None = None

    @property
    def is_trained(self) -> bool:
        """Whether :meth:`train` has produced a reusable consensus."""

        return bool(self.best_consensus)

    def _ensure_test_target_column(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """Ensure test data has at least two columns for LazyMIL compatibility."""

        df_test = df_test.copy()
        if len(df_test.columns) == 1:
            df_test[1] = [None for _ in df_test.index]
        return df_test


    def train(self, df_train: pd.DataFrame) -> MultiConformerModel:
        """Train/model-select once and cache everything required for later prediction.

        Args:
            df_train (pd.DataFrame): Training data; column 0 is SMILES,
                column 1 is the target.

        Returns:
            MultiConformerModel: This instance, now containing trained state.
        """

        train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=self.seed)
        lazy_ml = LazyMIL(
            num_conf=self.num_conf,
            hopt=self.hopt,
            num_cpu=self.num_cpu,
            output_folder=self.output_folder,
            verbose=self.verbose,
            seed=self.seed,
        )
        lazy_ml.run(train_df, val_df, val_df)

        res_val = pd.read_csv(f"{self.output_folder}/val.csv")
        x_val, true_val = res_val.iloc[:, 2:], res_val.iloc[:, 1]

        if self.verbose:
            print("\nRunning genetic consensus search ...")
            print(
                "Note: this step's randomness isn't controlled by `seed` - "
                "qsarcons.GeneticSearch doesn't expose one."
            )

        cons_search = GeneticSearch(cons_size="auto", n_iter=50)
        best_cons = cons_search.run(x_val, true_val)

        self.best_consensus = list(best_cons)
        self._consensus_search = cons_search
        self._lazy_model = lazy_ml

        if self.verbose:
            print("Best consensus:")
            print("\n".join(self.best_consensus))

        return self

    def predict(self, df_test: pd.DataFrame, save: bool = False) -> pd.DataFrame:
        """Predict for a new test dataframe using the stored trained state."""

        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call `train` or `load` first.")

        df_test = self._ensure_test_target_column(df_test)
        if self._lazy_model is not None and self._lazy_model.is_trained:
            res_test = self._lazy_model.predict(df_test, save=save)
        else:
            raise RuntimeError("LazyMIL model is not trained. Call `train` or `load` first.")

        x_test = res_test.iloc[:, 2:]

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

        pred_df = pd.concat([res_test["SMILES"], pd.Series(pred_test)], axis=1)
        pred_df = pred_df.rename(columns={0: "pred"})
        return pred_df

    def save(self, model_path: str | Path | None = None) -> None:
        """Serialize the trained model state to disk."""

        if not self.is_trained:
            raise RuntimeError("Model is not trained. Nothing to serialize.")

        model_path = Path(model_path or Path(self.output_folder) / "model.pkl")
        model_path.mkdir(parents=True, exist_ok=True)

        state = {
            "num_conf": self.num_conf,
            "hopt": self.hopt,
            "num_cpu": self.num_cpu,
            "verbose": self.verbose,
            "seed": self.seed,
            "best_consensus": self.best_consensus,
        }

        # Save metadata
        with model_path.joinpath("metadata.json").open("w") as f:
            json.dump(state, f, indent=4)

        # Save the underlying LazyMIL model
        if self._lazy_model is not None:
            models_dir = model_path / "models"
            self._lazy_model.save(models_dir / "consensus_model")


    @classmethod
    def load(cls, model_path: str | Path, output_folder: str | None = None) -> MultiConformerModel:
        """Load a serialized model state from the structured directory."""

        model_path = Path(model_path)
        metadata_file = model_path / "metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

        with metadata_file.open("r") as f:
            metadata = json.load(f)

        model = cls(
            num_conf=metadata["num_conf"],
            hopt=metadata["hopt"],
            num_cpu=metadata["num_cpu"],
            output_folder=output_folder,
            verbose=metadata["verbose"],
            seed=metadata["seed"],
        )
        model.best_consensus = list(metadata["best_consensus"])
        model._consensus_search = metadata.get("consensus_search")
        lazy_model_raw = metadata.get("lazy_model")
        if lazy_model_raw is not None:
            if isinstance(lazy_model_raw, LazyMIL):
                model._lazy_model = lazy_model_raw
            else:
                # Deserialized from JSON as a dict; reconstruct.
                lazy_path = model_path / "models" / "consensus_model"
                model._lazy_model = LazyMIL.load(str(lazy_path), output_folder=model.output_folder)
            if model._lazy_model is not None:
                model._lazy_model.output_folder = model.output_folder
                if os.path.exists(model.output_folder):
                    shutil.rmtree(model.output_folder)
                os.makedirs(model.output_folder)
        else:
            # lazy_model wasn't in metadata.json — try loading from disk
            lazy_path = model_path / "models" / "consensus_model"
            if lazy_path.exists():
                model._lazy_model = LazyMIL.load(str(lazy_path), output_folder=model.output_folder)
                model._lazy_model.output_folder = model.output_folder
                if os.path.exists(model.output_folder):
                    shutil.rmtree(model.output_folder)
                os.makedirs(model.output_folder)
            else:
                model._lazy_model = None
        return model

    def predictFromSMILES(self, smiles: list[str] | pd.Series) -> pd.DataFrame:
        """Load a serialized model and predict directly from SMILES strings."""

        df_test = pd.DataFrame({0: list(smiles)})
        return self.predict(df_test)

    def run_predict(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
        """Backwards-compatible one-shot API: train then predict.

        Prefer using :meth:`train`, :meth:`save`, :meth:`load`, and
        :meth:`predict` for a two-phase workflow.

        Args:
            df_train (pd.DataFrame): Training data; column 0 is SMILES,
                column 1 is the target.
            df_test (pd.DataFrame): Data to predict on; column 0 is SMILES.
                A target column is not required and is filled with ``None``
                if missing.

        Returns:
            pd.DataFrame: Two columns, ``SMILES`` and ``pred``, one row per
            test molecule.
        """

        self.train(df_train)
        return self.predict(df_test)

