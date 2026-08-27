from __future__ import annotations
# ruff: noqa: I001

from collections.abc import Callable, Iterable, Sequence
import os
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from milearn.network.classifier import (BagNetworkClassifier,
                                        InstanceNetworkClassifier,
                                        AdditiveAttentionNetworkClassifier,
                                        SelfAttentionNetworkClassifier,
                                        HopfieldAttentionNetworkClassifier,
                                        DynamicPoolingNetworkClassifier,
                                        )

from milearn.network.regressor import (BagNetworkRegressor,
                                       InstanceNetworkRegressor,
                                       AdditiveAttentionNetworkRegressor,
                                       SelfAttentionNetworkRegressor,
                                       HopfieldAttentionNetworkRegressor,
                                       DynamicPoolingNetworkRegressor,
                                        )

from milearn.network.regressor import InstanceWrapperMLPNetworkRegressor, BagWrapperMLPNetworkRegressor
from milearn.network.classifier import InstanceWrapperMLPNetworkClassifier, BagWrapperMLPNetworkClassifier

from qsarmil.conformer.rdkit import RDKitConformerGenerator
from qsarmil.descriptor.rdkit import RDKitAUTOCORR, RDKitGEOM, RDKitGETAWAY, RDKitMORSE, RDKitRDF, RDKitWHIM
from qsarmil.descriptor.wrapper import DescriptorWrapper
from molfeat.calc import ElectroShapeDescriptors, Pharmacophore3D, USRDescriptors

from milearn.preprocessing import BagMinMaxScaler
from rdkit import Chem, RDLogger

from qsarmil.utils.logging import FailedMolecule, OutputSuppressor


RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]


# ==========================================================
# Configuration
# ==========================================================

DESCRIPTORS: dict[str, Callable[[], DescriptorWrapper]] = {
    "RDKitGEOM": lambda: DescriptorWrapper(RDKitGEOM()),
    "RDKitAUTOCORR": lambda: DescriptorWrapper(RDKitAUTOCORR()),
    "RDKitRDF": lambda: DescriptorWrapper(RDKitRDF()),
    "RDKitMORSE": lambda: DescriptorWrapper(RDKitMORSE()),
    "RDKitWHIM": lambda: DescriptorWrapper(RDKitWHIM()),
    "MolFeatUSRD": lambda: DescriptorWrapper(USRDescriptors()),
    "MolFeatElectroShape": lambda: DescriptorWrapper(ElectroShapeDescriptors()),
    "RDKitGETAWAY": lambda: DescriptorWrapper(RDKitGETAWAY()),
    "MolFeatPmapper": lambda: DescriptorWrapper(Pharmacophore3D(factory="pmapper")),
}

REGRESSORS: dict[str, Callable[..., Any]] = {
    # mil wrappers
    "MeanInstanceWrapperMLPNetworkRegressor": lambda **kw: InstanceWrapperMLPNetworkRegressor(pool="mean", **kw),
    "MeanBagWrapperMLPNetworkRegressor": lambda **kw: BagWrapperMLPNetworkRegressor(pool="mean", **kw),
    # mil networks
    "MeanBagNetworkRegressor": lambda **kw: BagNetworkRegressor(pool="mean", **kw),
    "MeanInstanceNetworkRegressor": lambda **kw: InstanceNetworkRegressor(pool="mean", **kw),
    "AdditiveAttentionNetworkRegressor": lambda **kw: AdditiveAttentionNetworkRegressor(**kw),
    "SelfAttentionNetworkRegressor": lambda **kw: SelfAttentionNetworkRegressor(**kw),
    "HopfieldAttentionNetworkRegressor": lambda **kw: HopfieldAttentionNetworkRegressor(**kw),
    "DynamicPoolingNetworkRegressor": lambda **kw: DynamicPoolingNetworkRegressor(**kw),
}

CLASSIFIERS: dict[str, Callable[..., Any]] = {
    # mil wrappers
    "MeanInstanceWrapperMLPNetworkClassifier": lambda **kw: InstanceWrapperMLPNetworkClassifier(pool="mean", **kw),
    "MeanBagWrapperMLPNetworkClassifier": lambda **kw: BagWrapperMLPNetworkClassifier(pool="mean", **kw),
    # mil networks
    "MeanBagNetworkClassifier": lambda **kw: BagNetworkClassifier(pool="mean", **kw),
    "MeanInstanceNetworkClassifier": lambda **kw: InstanceNetworkClassifier(pool="mean", **kw),
    "AdditiveAttentionNetworkClassifier": lambda **kw: AdditiveAttentionNetworkClassifier(**kw),
    "SelfAttentionNetworkClassifier": lambda **kw: SelfAttentionNetworkClassifier(**kw),
    "HopfieldAttentionNetworkClassifier": lambda **kw: HopfieldAttentionNetworkClassifier(**kw),
    "DynamicPoolingNetworkClassifier": lambda **kw: DynamicPoolingNetworkClassifier(**kw),
}

HYPERPARAMETERS = {
    # Fixed hparams
    "max_epochs": 1000,
    "early_stopping": True,
    "accelerator": "cpu",
    "random_seed": 42,
    "verbose": False,
    "hidden_layer_sizes": [(2048, 1024, 512, 256, 128, 64), (256, 128, 64), (128,)],
    "activation": ["relu", "leakyrelu", "gelu", "elu", "silu"],
    "learning_rate": [1e-4, 1e-3],
}

# ==========================================================
# Pipeline steps
# ==========================================================

def generate_conformers(
    smi_list: Iterable[str],
    num_conf: int = 10,
    num_cpu: int = os.cpu_count() or 1,
    verbose: bool = False,
    random_seed: int = 42,
) -> list[list[Any] | FailedMolecule]:
    """Parse SMILES and generate conformers per molecule; failures become FailedMolecule sentinels, not raises."""

    mol_list = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        mol_list.append(mol if mol is not None else FailedMolecule(smi, message="SMILES parsing failed"))

    conf_gen = RDKitConformerGenerator(num_conf=num_conf, num_cpu=num_cpu, verbose=False, random_seed=random_seed)
    conf_list = conf_gen.run(mol_list)

    if verbose:
        n_valid = sum(isinstance(c, list) for c in conf_list)
        print(f"Generated conformers for {n_valid} of {len(conf_list)} molecules.")

    return conf_list


def calculate_descriptors(conf_list: list[list[Any]], calculator: DescriptorWrapper) -> list[np.ndarray]:
    """Compute descriptor bags for a list of per-molecule conformer bags."""
    calculator.verbose = False  # the low-level per-conformer ticker is redundant with LazyMIL's own step progress
    return calculator.run(conf_list)


def scale_descriptors(x_train: list[np.ndarray], x_test: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Min-max scale descriptor bags, fitting the scaler on the train set only; returns scaled descriptors."""
    scaler = BagMinMaxScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)


def baseline_prediction(y: Iterable[Any], task_type: str) -> Any:
    """Fallback prediction for molecules that can't be processed: the training mean or most frequent class."""

    y_arr = np.asarray(list(y))
    if task_type == "continuous":
        return float(np.mean(y_arr))
    values, counts = np.unique(y_arr, return_counts=True)
    return values[np.argmax(counts)]


# ==========================================================
# Estimator training
# ==========================================================
def train_estimator(
    x_train: list[np.ndarray],
    x_val: list[np.ndarray],
    x_test: list[np.ndarray],
    y_train: Iterable[Any],
    y_val: Iterable[Any],
    estimator: Any,
    hopt: bool = True,
    random_seed: int = 42,
    accelerator: str = "cpu",
) -> tuple[list[Any], list[Any], list[Any]]:
    """Fit one estimator, refit on train+val, and predict on train/val/test - all in one call, nothing persisted."""

    # 1. Scale train/val descriptors
    x_train_scaled, x_val_scaled = scale_descriptors(x_train, x_val)

    # 2. Optimize hyperparameters
    if hopt and hasattr(estimator, "hopt"):
        param_grid = {**HYPERPARAMETERS, "random_seed": random_seed, "accelerator": accelerator}
        estimator.hopt(x_train_scaled, y_train, param_grid=param_grid, verbose=False)

    # 3. Train on train split only (not final training yet)
    estimator.fit(x_train_scaled, y_train)
    pred_train = list(estimator.predict(x_train_scaled))
    pred_val = list(estimator.predict(x_val_scaled))

    # 4. Retrain on the full train+val set and predict on test with that same fit.
    x_full = x_train + x_val
    y_full = np.hstack((y_train, y_val))
    x_full_scaled, x_test_scaled = scale_descriptors(x_full, x_test)
    estimator.fit(x_full_scaled, y_full)
    pred_test = list(estimator.predict(x_test_scaled)) if x_test else []

    return pred_train, pred_val, pred_test


class LazyMIL:
    """Train every built-in descriptor/estimator combination and predict on train/val/test in one pass."""

    def __init__(
        self,
        task: str,
        hopt: bool = True,
        num_conf: int = 10,
        num_cpu: int = os.cpu_count() or 1,
        output_folder: str | None = None,
        verbose: bool = True,
        random_seed: int = 42,
        accelerator: str = "cpu",
    ) -> None:
        """Store settings and (re)create the output folder."""
        self.task = task
        self.ESTIMATORS = REGRESSORS if task == "continuous" else CLASSIFIERS
        self.hopt = hopt
        self.num_conf = num_conf
        self.output_folder: str = output_folder or tempfile.mkdtemp(prefix="qsarmil_")
        self.num_cpu = num_cpu
        self.verbose = verbose
        self.random_seed = random_seed
        self.accelerator = accelerator

        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)

    def run(
        self,
        smiles_train: Sequence[str],
        y_train: Sequence[Any],
        smiles_val: Sequence[str],
        y_val: Sequence[Any],
        smiles_test: Sequence[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Train every descriptor/estimator combination and predict on train/val/test, writing CSVs as it goes."""

        smi_train, y_train = list(smiles_train), list(y_train)
        smi_val, y_val = list(smiles_val), list(y_val)
        smi_test = list(smiles_test)

        # 1. Parse SMILES and generate conformers.
        if self.verbose:
            print("Step-1. Conformer generation")
        smi_all = smi_train + smi_val + smi_test
        conf_all = generate_conformers(
            smi_all, num_conf=self.num_conf, num_cpu=self.num_cpu, verbose=self.verbose,
            random_seed=self.random_seed,
        )
        n_train, n_val = len(smi_train), len(smi_val)
        conf_train = conf_all[:n_train]
        conf_val = conf_all[n_train : n_train + n_val]
        conf_test = conf_all[n_train + n_val :]

        # Train/val: drop molecules that failed - training needs clean data.
        valid_idx_train = [i for i, c in enumerate(conf_train) if isinstance(c, list)]
        smi_train = [smi_train[i] for i in valid_idx_train]
        y_train = [y_train[i] for i in valid_idx_train]
        conf_train = [conf_train[i] for i in valid_idx_train]

        valid_idx_val = [i for i, c in enumerate(conf_val) if isinstance(c, list)]
        smi_val = [smi_val[i] for i in valid_idx_val]
        y_val = [y_val[i] for i in valid_idx_val]
        conf_val = [conf_val[i] for i in valid_idx_val]

        # Keep all test molecules - failures get the training set baseline instead of a real prediction.
        train_baseline = baseline_prediction(y_train, self.task)
        valid_idx_test = [i for i, c in enumerate(conf_test) if isinstance(c, list)]
        smi_test_valid = [smi_test[i] for i in valid_idx_test]
        conf_test_valid = [conf_test[i] for i in valid_idx_test]

        n_failed_test = len(smi_test) - len(smi_test_valid)
        if n_failed_test and self.verbose:
            print(
                f"{n_failed_test} test molecule(s) could not be processed and will be predicted "
                "using the training set baseline value instead."
            )

        result_df_train = pd.DataFrame({"SMILES": smi_train, "Y_TRUE": y_train})
        result_df_val = pd.DataFrame({"SMILES": smi_val, "Y_TRUE": y_val})
        result_df_test = pd.DataFrame({"SMILES": smi_test})

        # 2. Calculate descriptors for train+val+test together, once per descriptor type.
        if self.verbose:
            print("Step-2. Descriptor calculation")

        ready_descriptors: dict[str, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]] = {}
        for desc_name, desc_factory in DESCRIPTORS.items():
            desc_calc = desc_factory()
            x_all = calculate_descriptors(conf_train + conf_val + conf_test_valid, desc_calc)
            x_train = x_all[: len(conf_train)]
            x_val = x_all[len(conf_train) : len(conf_train) + len(conf_val)]
            x_test = x_all[len(conf_train) + len(conf_val) :]
            ready_descriptors[desc_name] = (x_train, x_val, x_test)

            if self.verbose:
                print(f"{desc_name}: done")

        # 3. Train every descriptor/estimator combination and predict on train/val/test.
        if self.verbose:
            print("Step-3. Model training")

        total_models = len(DESCRIPTORS) * len(self.ESTIMATORS)
        current_model = 0

        for desc_name, (x_train, x_val, x_test) in ready_descriptors.items():
            for est_name, est_factory in self.ESTIMATORS.items():
                estimator = est_factory(accelerator=self.accelerator)
                model_name = f"{desc_name}|{est_name}"
                current_model += 1

                with OutputSuppressor():
                    pred_train, pred_val, pred_test = train_estimator(
                        x_train, x_val, x_test, y_train, y_val, estimator, self.hopt,
                        random_seed=self.random_seed, accelerator=self.accelerator,
                    )

                preds_by_smi = dict(zip(smi_test_valid, pred_test))

                result_df_train[model_name] = pred_train
                result_df_val[model_name] = pred_val
                result_df_test[model_name] = [preds_by_smi.get(smi, train_baseline) for smi in smi_test]

                result_df_train.to_csv(os.path.join(self.output_folder, "train.csv"), index=False)
                result_df_val.to_csv(os.path.join(self.output_folder, "val.csv"), index=False)
                result_df_test.to_csv(os.path.join(self.output_folder, "test.csv"), index=False)

                if self.verbose:
                    print(f"[{current_model}/{total_models}] {model_name}")

        return result_df_train, result_df_val, result_df_test
