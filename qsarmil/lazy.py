from __future__ import annotations
# ruff: noqa: I001

from collections.abc import Callable, Iterable, Mapping
import os
import pickle
import shutil
import tempfile
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from milearn.preprocessing import BagMinMaxScaler
from rdkit import Chem, RDLogger
from sklearn.utils.multiclass import type_of_target

from qsarmil.conformer.rdkit import RDKitConformerGenerator
from qsarmil.data.input_data import DataValidator
from qsarmil.descriptor.wrapper import DescriptorWrapper
from qsarmil.utils.ensemble import ConformerEnsemble

from qsarmil.utils.logging import OutputSuppressor

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

# ==========================================================
# Configuration
# ==========================================================
def _DESCRIPTORS() -> dict[str, Callable[[], DescriptorWrapper]]:
    def factory(module_name: str, class_name: str, /, *args: Any, **kwargs: Any) -> Callable[[], DescriptorWrapper]:
        def build() -> DescriptorWrapper:
            cls = getattr(import_module(module_name), class_name)
            return DescriptorWrapper(cls(*args, **kwargs))

        return build

    return {
        "RDKitGEOM": factory("qsarmil.descriptor.rdkit", "RDKitGEOM"),
        "RDKitAUTOCORR": factory("qsarmil.descriptor.rdkit", "RDKitAUTOCORR"),
        "RDKitRDF": factory("qsarmil.descriptor.rdkit", "RDKitRDF"),
        "RDKitMORSE": factory("qsarmil.descriptor.rdkit", "RDKitMORSE"),
        "RDKitWHIM": factory("qsarmil.descriptor.rdkit", "RDKitWHIM"),
        "MolFeatUSRD": factory("molfeat.calc", "USRDescriptors"),
        "MolFeatElectroShape": factory("molfeat.calc", "ElectroShapeDescriptors"),
        "RDKitGETAWAY": factory("qsarmil.descriptor.rdkit", "RDKitGETAWAY"),
        "MolFeatPmapper": factory("molfeat.calc", "Pharmacophore3D", factory="pmapper"),
    }


DESCRIPTORS = _DESCRIPTORS()

def _REGRESSORS() -> dict[str, Any]:
    def factory(module_name: str, class_name: str, /, *args: Any, **kwargs: Any) -> Any:
        def build() -> Any:
            cls = getattr(import_module(module_name), class_name)
            return cls(*args, **kwargs)

        return build

    return {
        # mil wrappers
        "MeanInstanceWrapperMLPNetworkRegressor": factory(
            "milearn.network.regressor", "InstanceWrapperMLPNetworkRegressor", pool="mean"
        ),
        "MeanBagWrapperMLPNetworkRegressor": factory(
            "milearn.network.regressor", "BagWrapperMLPNetworkRegressor", pool="mean"
        ),
        # mil networks
        "MeanBagNetworkRegressor": factory("milearn.network.regressor", "BagNetworkRegressor", pool="mean"),
        "MeanInstanceNetworkRegressor": factory("milearn.network.regressor", "InstanceNetworkRegressor", pool="mean"),
        "AdditiveAttentionNetworkRegressor": factory("milearn.network.regressor", "AdditiveAttentionNetworkRegressor"),
        "SelfAttentionNetworkRegressor": factory("milearn.network.regressor", "SelfAttentionNetworkRegressor"),
        "HopfieldAttentionNetworkRegressor": factory("milearn.network.regressor", "HopfieldAttentionNetworkRegressor"),
        "DynamicPoolingNetworkRegressor": factory("milearn.network.regressor", "DynamicPoolingNetworkRegressor"),
        # classical
        "Ridge": factory("sklearn.linear_model", "Ridge"),
        "MLPRegressor": factory("sklearn.neural_network", "MLPRegressor"),
        "LinearSVR": factory("sklearn.svm", "LinearSVR"),
        "XGBRegressor": factory("xgboost", "XGBRegressor"),
    }


def _CLASSIFIERS() -> dict[str, Any]:
    def factory(module_name: str, class_name: str, /, *args: Any, **kwargs: Any) -> Any:
        def build() -> Any:
            cls = getattr(import_module(module_name), class_name)
            return cls(*args, **kwargs)

        return build

    return {
        # mil wrappers
        "MeanInstanceWrapperMLPNetworkClassifier": factory(
            "milearn.network.classifier", "InstanceWrapperMLPNetworkClassifier", pool="mean"
        ),
        "MeanBagWrapperMLPNetworkClassifier": factory(
            "milearn.network.classifier", "BagWrapperMLPNetworkClassifier", pool="mean"
        ),
        # mil networks
        "MeanBagNetworkClassifier": factory("milearn.network.classifier", "BagNetworkClassifier", pool="mean"),
        "MeanInstanceNetworkClassifier": factory("milearn.network.classifier", "InstanceNetworkClassifier", pool="mean"),
        "AdditiveAttentionNetworkClassifier": factory("milearn.network.classifier", "AdditiveAttentionNetworkClassifier"),
        "SelfAttentionNetworkClassifier": factory("milearn.network.classifier", "SelfAttentionNetworkClassifier"),
        "HopfieldAttentionNetworkClassifier": factory("milearn.network.classifier", "HopfieldAttentionNetworkClassifier"),
        "DynamicPoolingNetworkClassifier": factory("milearn.network.classifier", "DynamicPoolingNetworkClassifier"),
        # classical
        "RidgeClassifier": factory("sklearn.linear_model", "RidgeClassifier"),
        "MLPClassifier": factory("sklearn.neural_network", "MLPClassifier"),
        "LinearSVC": factory("sklearn.svm", "LinearSVC"),
        "XGBClassifier": factory("xgboost", "XGBClassifier"),
    }

# Lazy estimator mappings. The dictionaries are built eagerly, but each value
# remains a zero-argument factory so the actual estimator import/instantiation
# only happens when ``factory()`` is reached during training.
REGRESSORS = _REGRESSORS()
CLASSIFIERS = _CLASSIFIERS()

DEFAULT_PARAM_GRID = {
    # Fixed hparams
    "max_epochs": 1000,
    "early_stopping": True,
    "accelerator": "cpu",
    "random_seed": 42,
    "verbose": False,
    "hidden_layer_sizes": [(2048, 1024, 512, 256, 128, 64), (256, 128, 64), (128,)],
    "activation": ["relu", "leakyrelu", "gelu", "elu", "silu"],
    "learning_rate": [10e-5, 10e-4],
}


def resolve_estimators(estimators: Callable[[], Mapping[str, Any]] | Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve estimator sources that may be either a lazy factory or a ready mapping."""

    return estimators() if callable(estimators) else estimators

# ==========================================================
# Utility Functions
# ==========================================================

def gen_conformers(
    smi_list: Iterable[str], num_conf: int = 10, num_cpu: int = 1, verbose: bool = False, seed: int = 42
) -> list[ConformerEnsemble]:
    """Generate conformers for a list of SMILES strings using
    RDKitConformerGenerator."""
    mol_list = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        mol_list.append(mol)
    conf_gen = RDKitConformerGenerator(num_conf=num_conf, num_cpu=num_cpu, verbose=verbose, seed=seed)
    conf_list = conf_gen.run(mol_list)
    return conf_list

def compute_column_means(bags: list[np.ndarray]) -> np.ndarray:
    """Compute per-column, NaN-ignoring means across every instance in every bag."""

    all_instances = np.vstack(bags)
    return np.nanmean(all_instances, axis=0)

def clean_descriptors(bags: list[np.ndarray], col_means: np.ndarray | None = None) -> list[np.ndarray]:
    """Replace NaN values in each bag's instances with column means.

    Args:
        bags (list[np.ndarray]): Descriptor bags to clean.
        col_means (np.ndarray, optional): Per-column means to impute with.
            If omitted, computed from ``bags`` themselves.

    Returns:
        list[np.ndarray]: Cleaned descriptor bags.
    """

    if col_means is None:
        col_means = compute_column_means(bags)

    # Replace NaNs in each bag with the corresponding column mean
    cleaned_bags = []
    for bag in bags:
        bag = np.array(bag, dtype=float)  # Ensure float for NaN support
        idx = np.where(np.isnan(bag))
        bag[idx] = np.take(col_means, idx[1])
        cleaned_bags.append(bag)

    return cleaned_bags

def calc_descriptors(
    conf_list: list[ConformerEnsemble],
    calculator: DescriptorWrapper,
    verbose: bool = False,
    col_means: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Compute and NaN-clean descriptor bags for a list of conformer ensembles.

    Args:
        conf_list (list[ConformerEnsemble]): Per-molecule conformer ensembles.
        calculator (DescriptorWrapper): Descriptor calculator to apply.
        verbose (bool): Whether the calculator should print progress.
        col_means (np.ndarray, optional): Column means to impute NaNs with.
            If omitted, means are computed from this call's own output.

    Returns:
        list[np.ndarray]: One cleaned descriptor bag per molecule. Assumes
        descriptor calculation succeeded for every molecule; a
        :class:`~qsarmil.utils.logging.FailedDescriptor` in ``calculator``'s
        output is not handled here and will raise inside
        :func:`clean_descriptors` instead.
    """
    calculator.verbose = verbose
    x: list[Any] = calculator.run(conf_list)
    x = clean_descriptors(x, col_means=col_means)
    return x

def scale_descriptors(x_train: list[np.ndarray], x_test: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Min-max scale descriptor bags, fitting the scaler on the train set only.

    Args:
        x_train (list[np.ndarray]): Training bags to fit the scaler on.
        x_test (list[np.ndarray]): Bags to scale using that same fit.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: Scaled ``(x_train, x_test)``.
    """
    scaler = BagMinMaxScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)


def _pool_bags_mean(x_bags: list[np.ndarray]) -> np.ndarray:
    """Pool each bag to one feature vector via instance-wise mean."""

    return np.vstack([np.nanmean(np.asarray(bag, dtype=float), axis=0) for bag in x_bags])


def _estimator_supports_bag_input(estimator_instance: Any) -> bool:
    """Heuristic: sklearn/xgboost estimators typically require 2D matrix input."""

    module_name = estimator_instance.__class__.__module__
    return not (module_name.startswith("sklearn.") or module_name.startswith("xgboost."))


def _prepare_features_for_estimator(
    estimator_instance: Any,
    x_train: list[np.ndarray],
    x_val: list[np.ndarray],
    x_test: list[np.ndarray],
) -> tuple[Any, Any, Any]:
    """Return either bag inputs or pooled 2D inputs depending on estimator type."""

    if _estimator_supports_bag_input(estimator_instance):
        return x_train, x_val, x_test
    return _pool_bags_mean(x_train), _pool_bags_mean(x_val), _pool_bags_mean(x_test)

# ==========================================================
# ModelBuilder Class
# ==========================================================
def build_model(
    x_train: list[np.ndarray],
    x_val: list[np.ndarray],
    x_test: list[np.ndarray],
    y_train: Iterable[Any],
    y_val: Iterable[Any],
    y_test: Iterable[Any],
    estimator_instance: Any,
    hopt: bool = True,
    seed: int = 42,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Fit one estimator and return its predictions on train/val/test.

    Tunes hyperparameters (if requested and supported) and fits on the
    train split to produce train/val predictions, then refits the same
    estimator on train+val combined to produce the final test predictions.

    Args:
        x_train (list[np.ndarray]): Training descriptor bags.
        x_val (list[np.ndarray]): Validation descriptor bags.
        x_test (list[np.ndarray]): Test descriptor bags.
        y_train (array-like): Training targets.
        y_val (array-like): Validation targets.
        y_test (array-like): Test targets (unused, kept for signature symmetry).
        estimator_instance: A MIL estimator implementing ``fit``/``predict``,
            and optionally ``hopt`` for hyperparameter search.
        hopt (bool): Whether to run ``estimator_instance.hopt`` before fitting,
            if the estimator supports it.
        seed (int): Random seed passed to the estimator's hyperparameter
            search, overriding ``DEFAULT_PARAM_GRID``'s own default.

    Returns:
        tuple[list, list, list]: ``(pred_train, pred_val, pred_test)``.
    """

    # 1. Scale train/val descriptors
    x_train_scaled, x_val_scaled = scale_descriptors(x_train, x_val)

    fit_x_train, fit_x_val, _ = _prepare_features_for_estimator(
        estimator_instance, x_train_scaled, x_val_scaled, x_val_scaled
    )

    # 2. Optimize hyperparameters
    if hopt and hasattr(estimator_instance, "hopt"):
        param_grid = {**DEFAULT_PARAM_GRID, "random_seed": seed}
        estimator_instance.hopt(fit_x_train, y_train, param_grid=param_grid, verbose=False)

    # 4. Train on train split only (not final training yet)
    estimator_instance.fit(fit_x_train, y_train)
    pred_train = list(estimator_instance.predict(fit_x_train))
    pred_val = list(estimator_instance.predict(fit_x_val))

    # 5. Retrain model on full (train + val)
    x_full, y_full = x_train + x_val, np.hstack((y_train, y_val))
    x_full_scaled, x_test_scaled = scale_descriptors(x_full, x_test)
    fit_x_full, _, fit_x_test_scaled = _prepare_features_for_estimator(
        estimator_instance, x_full_scaled, x_full_scaled, x_test_scaled
    )
    estimator_instance.fit(fit_x_full, y_full)
    pred_test = list(estimator_instance.predict(fit_x_test_scaled))

    return pred_train, pred_val, pred_test


def build_model_with_artifacts(
    x_train: list[np.ndarray],
    x_val: list[np.ndarray],
    x_test: list[np.ndarray],
    y_train: Iterable[Any],
    y_val: Iterable[Any],
    y_test: Iterable[Any],
    estimator_instance: Any,
    hopt: bool = True,
    seed: int = 42,
) -> tuple[list[Any], list[Any], list[Any], Any, BagMinMaxScaler]:
    """Fit one estimator, return predictions and the final fitted artifacts.

    The returned estimator/scaler pair is fitted on train+val and can be
    serialized for test-time inference without retraining.
    """

    # 1. Scale train/val descriptors
    x_train_scaled, x_val_scaled = scale_descriptors(x_train, x_val)

    fit_x_train, fit_x_val, _ = _prepare_features_for_estimator(
        estimator_instance, x_train_scaled, x_val_scaled, x_val_scaled
    )

    # 2. Optimize hyperparameters
    if hopt and hasattr(estimator_instance, "hopt"):
        param_grid = {**DEFAULT_PARAM_GRID, "random_seed": seed}
        estimator_instance.hopt(fit_x_train, y_train, param_grid=param_grid, verbose=False)

    # 3. Train on train split for train/val predictions
    estimator_instance.fit(fit_x_train, y_train)
    pred_train = list(estimator_instance.predict(fit_x_train))
    pred_val = list(estimator_instance.predict(fit_x_val))

    # 4. Retrain on full (train + val) for deployable test-time inference
    x_full, y_full = x_train + x_val, np.hstack((y_train, y_val))
    scaler_full = BagMinMaxScaler()
    scaler_full.fit(x_full)
    x_full_scaled = scaler_full.transform(x_full)
    x_test_scaled = scaler_full.transform(x_test)
    fit_x_full, _, fit_x_test_scaled = _prepare_features_for_estimator(
        estimator_instance, x_full_scaled, x_full_scaled, x_test_scaled
    )
    estimator_instance.fit(fit_x_full, y_full)
    pred_test = list(estimator_instance.predict(fit_x_test_scaled))

    return pred_train, pred_val, pred_test, estimator_instance, scaler_full


class LazyMIL:
    """Train every combination of built-in descriptor and MIL estimator on one dataset.

    For each of the 9 built-in 3D descriptor types crossed with every
    regressor or classifier in :data:`REGRESSORS`/:data:`CLASSIFIERS` (task
    type is inferred from the training targets), generates conformers,
    computes descriptors, fits the estimator, and writes predictions to
    ``train.csv``/``val.csv``/``test.csv`` under ``output_folder``.
    """

    def __init__(
        self,
        hopt: bool = True,
        num_conf: int = 10,
        num_cpu: int = 20,
        output_folder: str | None = None,
        verbose: bool = True,
        seed: int = 42,
    ) -> None:
        """Set up the run and (re)create the output folder.

        Args:
            hopt (bool): Whether to hyperparameter-tune each estimator before
                fitting, for estimators that support it.
            num_conf (int): Number of conformers to generate per molecule.
            num_cpu (int): Number of CPU threads to use for conformer generation.
            output_folder (str, optional): Directory the per-model prediction
                CSVs are written to. If omitted, a fresh temporary directory
                is created. If it already exists, it's wiped and recreated.
            verbose (bool): Whether to print per-model progress and memory usage.
            seed (int): Random seed used for conformer embedding, molecule
                validation, and hyperparameter search.
        """
        self.hopt = hopt
        self.num_conf = num_conf
        self.output_folder: str = output_folder or tempfile.mkdtemp(prefix="qsarmil_")
        self.num_cpu = num_cpu
        self.verbose = verbose
        self.seed = seed

        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)

        # Populated after ``run`` and reused by ``predict`` for inference-only
        # execution (descriptor generation + estimator.predict only).
        self._trained_models: dict[str, dict[str, Any]] = {}
        self._task_type: str | None = None

    @property
    def is_trained(self) -> bool:
        """Whether this instance has serialized-ready fitted model artifacts."""

        return bool(self._trained_models)

    def run(self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
        """Train every descriptor/estimator combination and write predictions to CSV.

        Args:
            df_train (pd.DataFrame): Training data; column 0 is SMILES,
                column 1 is the target. Rows with unparseable or
                non-embeddable SMILES are dropped before training.
            df_val (pd.DataFrame): Validation data, same column layout and filtering.
            df_test (pd.DataFrame): Test data, same column layout and filtering.

        Returns:
            None. Results are written to ``train.csv``, ``val.csv`` and
            ``test.csv`` in ``self.output_folder``, with one prediction
            column per descriptor/estimator combination.
        """

        # Reset previous fitted artifacts for a fresh training run.
        self._trained_models = {}

        # 1. Drop molecules that don't parse/sanitize/embed in 3D, so one
        #    bad SMILES doesn't crash the whole run
        validator = DataValidator(num_cpu=self.num_cpu, verbose=self.verbose, seed=self.seed)
        df_train = validator.filter_dataframe(df_train)
        df_val = validator.filter_dataframe(df_val)
        df_test = validator.filter_dataframe(df_test)

        # 2. Get data (smiles and prop)
        result_df_train = pd.DataFrame()
        smi_train, y_train = list(df_train.iloc[:, 0]), list(df_train.iloc[:, 1])
        result_df_train["SMILES"], result_df_train["Y_TRUE"] = smi_train, y_train

        result_df_val = pd.DataFrame()
        smi_val, y_val = list(df_val.iloc[:, 0]), list(df_val.iloc[:, 1])
        result_df_val["SMILES"], result_df_val["Y_TRUE"] = smi_val, y_val

        result_df_test = pd.DataFrame()
        smi_test, y_test = list(df_test.iloc[:, 0]), list(df_test.iloc[:, 1])
        result_df_test["SMILES"], result_df_test["Y_TRUE"] = smi_test, y_test

        # 3. Get a task type
        task_type = type_of_target(y_train)
        if task_type == "continuous":
            estimators_source = REGRESSORS
        elif task_type == "binary":
            estimators_source = CLASSIFIERS
        else:
            raise ValueError(
                f"Task type '{task_type}' not supported (only 'continuous' and 'binary' targets are supported)."
            )
        self._task_type = task_type

        # 4. Generate conformers
        conf_train = gen_conformers(
            smi_train, num_conf=self.num_conf, num_cpu=self.num_cpu, verbose=self.verbose, seed=self.seed
        )
        conf_val = gen_conformers(
            smi_val, num_conf=self.num_conf, num_cpu=self.num_cpu, verbose=self.verbose, seed=self.seed
        )
        conf_test = gen_conformers(
            smi_test, num_conf=self.num_conf, num_cpu=self.num_cpu, verbose=self.verbose, seed=self.seed
        )

        total_models = len(DESCRIPTORS) * len(estimators_source)
        current_model = 0

        # 5. Calculate descriptors, imputing val/test NaNs with train's own
        #    column means
        for desc_name, desc_source in DESCRIPTORS.items():
            desc_calc = desc_source()

            x_train = list(calc_descriptors(conf_train, desc_calc, verbose=False))
            train_col_means = compute_column_means(x_train)
            x_val = list(calc_descriptors(conf_val, desc_calc, verbose=False, col_means=train_col_means))
            x_test = list(calc_descriptors(conf_test, desc_calc, verbose=False, col_means=train_col_means))

            # 6. Train models
            for est_name, factory in estimators_source.items():
                estimator = factory()

                model_name = f"{desc_name}|{est_name}"
                current_model += 1

                start = time.time()
                with OutputSuppressor():
                    pred_train, pred_val, pred_test, fitted_estimator, fitted_scaler = build_model_with_artifacts(
                        x_train, x_val, x_test, y_train, y_val, y_test, estimator, self.hopt, seed=self.seed
                    )
                elapsed_min = (time.time() - start) / 60

                # Persist everything needed for inference-only execution.
                self._trained_models[model_name] = {
                    "descriptor": desc_name,
                    "estimator": fitted_estimator,
                    "scaler": fitted_scaler,
                    "train_col_means": train_col_means,
                }

                # 7. Write predictions
                result_df_train[model_name] = pred_train
                result_df_train.to_csv(os.path.join(self.output_folder, "train.csv"), index=False)

                result_df_val[model_name] = pred_val
                result_df_val.to_csv(os.path.join(self.output_folder, "val.csv"), index=False)

                result_df_test[model_name] = pred_test
                result_df_test.to_csv(os.path.join(self.output_folder, "test.csv"), index=False)

                if self.verbose:
                    process = psutil.Process()
                    mem_gb = process.memory_info().rss / (1024**3)
                    print(f"[{current_model}/{total_models}] Running model: {model_name}")
                    print(f"  > Finished in {elapsed_min:.2f} min | Memory usage: {mem_gb:.3f} GB")

    def predict(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """Run inference from persisted fitted models without retraining.

        This path only validates SMILES, generates conformers/descriptors,
        scales descriptors with stored scalers, and calls estimator.predict.
        """

        if not self.is_trained:
            raise RuntimeError("LazyMIL is not trained. Call `run` or `load` first.")

        df_test = df_test.copy()
        if len(df_test.columns) == 1:
            df_test[1] = [None for _ in df_test.index]

        validator = DataValidator(num_cpu=self.num_cpu, verbose=self.verbose, seed=self.seed)
        df_test = validator.filter_dataframe(df_test)

        result_df_test = pd.DataFrame()
        smi_test, y_test = list(df_test.iloc[:, 0]), list(df_test.iloc[:, 1])
        result_df_test["SMILES"], result_df_test["Y_TRUE"] = smi_test, y_test

        conf_test = gen_conformers(
            smi_test, num_conf=self.num_conf, num_cpu=self.num_cpu, verbose=self.verbose, seed=self.seed
        )

        descriptor_cache: dict[str, list[np.ndarray]] = {}
        descriptor_means: dict[str, np.ndarray] = {}
        for model_state in self._trained_models.values():
            descriptor_means[model_state["descriptor"]] = model_state["train_col_means"]

        for desc_name, col_means in descriptor_means.items():
            if desc_name not in DESCRIPTORS:
                raise ValueError(
                    f"Descriptor '{desc_name}' was used during training but isn't available in current DESCRIPTORS."
                )
            desc_calc = DESCRIPTORS[desc_name]()
            descriptor_cache[desc_name] = list(
                calc_descriptors(conf_test, desc_calc, verbose=False, col_means=col_means)
            )

        for model_name, model_state in self._trained_models.items():
            desc_name = model_state["descriptor"]
            scaler = model_state["scaler"]
            estimator = model_state["estimator"]
            x_test_scaled = scaler.transform(descriptor_cache[desc_name])
            _, _, fit_x_test_scaled = _prepare_features_for_estimator(
                estimator, x_test_scaled, x_test_scaled, x_test_scaled
            )
            result_df_test[model_name] = list(estimator.predict(fit_x_test_scaled))

        result_df_test.to_csv(os.path.join(self.output_folder, "test.csv"), index=False)
        return result_df_test

    def save(self, model_path: str | Path) -> None:
        """Serialize fitted artifacts so future inference skips retraining."""

        if not self.is_trained:
            raise RuntimeError("LazyMIL is not trained. Nothing to serialize.")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "hopt": self.hopt,
            "num_conf": self.num_conf,
            "num_cpu": self.num_cpu,
            "verbose": self.verbose,
            "seed": self.seed,
            "task_type": self._task_type,
            "trained_models": self._trained_models,
        }
        with model_path.open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, model_path: str | Path, output_folder: str | None = None) -> LazyMIL:
        """Load a serialized LazyMIL artifact for inference-only use."""

        model_path = Path(model_path)
        with model_path.open("rb") as f:
            state = pickle.load(f)

        model = cls(
            hopt=state["hopt"],
            num_conf=state["num_conf"],
            num_cpu=state["num_cpu"],
            output_folder=output_folder,
            verbose=state["verbose"],
            seed=state["seed"],
        )
        model._task_type = state.get("task_type")
        model._trained_models = state["trained_models"]
        return model

