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

from qsarmil.utils.logging import FailedConformer, FailedMolecule, OutputSuppressor


RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]


# ==========================================================
# Configuration
# ==========================================================
# Every entry is a zero-argument factory rather than a ready-made instance, so that each call to
# LazyMIL.run() gets fresh, independent DescriptorWrapper objects instead of sharing (and mutating)
# module-level state across runs.
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

# Same reasoning as DESCRIPTORS above: factories, not instances, so every run() call and every
# descriptor gets its own fresh estimator rather than 9 descriptors overwriting one shared model.
# accelerator is threaded through explicitly (default "cpu") so LazyMIL can pass its own setting in.
REGRESSORS: dict[str, Callable[..., Any]] = {
    # mil wrappers
    "MeanInstanceWrapperMLPNetworkRegressor": lambda accelerator="cpu": InstanceWrapperMLPNetworkRegressor(
        pool="mean", accelerator=accelerator
    ),
    "MeanBagWrapperMLPNetworkRegressor": lambda accelerator="cpu": BagWrapperMLPNetworkRegressor(
        pool="mean", accelerator=accelerator
    ),
    # mil networks
    "MeanBagNetworkRegressor": lambda accelerator="cpu": BagNetworkRegressor(pool="mean", accelerator=accelerator),
    "MeanInstanceNetworkRegressor": lambda accelerator="cpu": InstanceNetworkRegressor(
        pool="mean", accelerator=accelerator
    ),
    "AdditiveAttentionNetworkRegressor": lambda accelerator="cpu": AdditiveAttentionNetworkRegressor(
        accelerator=accelerator
    ),
    "SelfAttentionNetworkRegressor": lambda accelerator="cpu": SelfAttentionNetworkRegressor(
        accelerator=accelerator
    ),
    "HopfieldAttentionNetworkRegressor": lambda accelerator="cpu": HopfieldAttentionNetworkRegressor(
        accelerator=accelerator
    ),
    "DynamicPoolingNetworkRegressor": lambda accelerator="cpu": DynamicPoolingNetworkRegressor(
        accelerator=accelerator
    ),
}

CLASSIFIERS: dict[str, Callable[..., Any]] = {
    # mil wrappers
    "MeanInstanceWrapperMLPNetworkClassifier": lambda accelerator="cpu": InstanceWrapperMLPNetworkClassifier(
        pool="mean", accelerator=accelerator
    ),
    "MeanBagWrapperMLPNetworkClassifier": lambda accelerator="cpu": BagWrapperMLPNetworkClassifier(
        pool="mean", accelerator=accelerator
    ),
    # mil networks
    "MeanBagNetworkClassifier": lambda accelerator="cpu": BagNetworkClassifier(pool="mean", accelerator=accelerator),
    "MeanInstanceNetworkClassifier": lambda accelerator="cpu": InstanceNetworkClassifier(
        pool="mean", accelerator=accelerator
    ),
    "AdditiveAttentionNetworkClassifier": lambda accelerator="cpu": AdditiveAttentionNetworkClassifier(
        accelerator=accelerator
    ),
    "SelfAttentionNetworkClassifier": lambda accelerator="cpu": SelfAttentionNetworkClassifier(
        accelerator=accelerator
    ),
    "HopfieldAttentionNetworkClassifier": lambda accelerator="cpu": HopfieldAttentionNetworkClassifier(
        accelerator=accelerator
    ),
    "DynamicPoolingNetworkClassifier": lambda accelerator="cpu": DynamicPoolingNetworkClassifier(
        accelerator=accelerator
    ),
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

def parse_smiles(smi_list: Iterable[str], verbose: bool = False) -> list[Any]:
    """Parse SMILES strings into RDKit molecules; unparseable ones become FailedMolecule sentinels.

    Args:
        smi_list (Iterable[str]): SMILES strings.
        verbose (bool): Whether to print a one-line success/failure summary.

    Returns:
        list: One RDKit ``Mol`` or :class:`~qsarmil.utils.logging.FailedMolecule` per input SMILES.
    """

    mol_list = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        mol_list.append(mol if mol is not None else FailedMolecule(smi))

    if verbose:
        n_failed = sum(isinstance(m, FailedMolecule) for m in mol_list)
        print(f"Parsed {len(mol_list) - n_failed} of {len(mol_list)} SMILES successfully.")

    return mol_list


def generate_conformers(
    mol_list: Iterable[Any],
    num_conf: int = 10,
    num_cpu: int = os.cpu_count() or 1,
    verbose: bool = False,
    seed: int = 42,
) -> list[list[Any] | FailedMolecule | FailedConformer]:
    """Generate conformers per molecule; failed embeddings become FailedConformer sentinels, not raises.

    Args:
        mol_list (Iterable[Any]): RDKit molecules, or :class:`~qsarmil.utils.logging.FailedMolecule`
            sentinels (from :func:`parse_smiles`), which pass through unchanged.
        num_conf (int): Number of conformers to embed per molecule.
        num_cpu (int): Number of threads to use for conformer generation.
        verbose (bool): Whether to print a one-line success/failure summary.
        seed (int): Random seed for conformer embedding.

    Returns:
        list: One conformer bag (``list[Mol]``) per successfully-embedded molecule, or a
            :class:`~qsarmil.utils.logging.FailedMolecule`/:class:`~qsarmil.utils.logging.FailedConformer`
            sentinel for the rest.
    """

    conf_gen = RDKitConformerGenerator(num_conf=num_conf, num_cpu=num_cpu, verbose=False, seed=seed)
    conf_list = conf_gen.run(list(mol_list))

    if verbose:
        n_ok = sum(isinstance(c, list) for c in conf_list)
        print(f"Generated conformers for {n_ok} of {len(conf_list)} molecules.")

    return conf_list


def calculate_descriptors(conf_list: list[list[Any]], calculator: DescriptorWrapper) -> list[np.ndarray]:
    """Compute descriptor bags for a list of per-molecule conformer bags."""
    calculator.verbose = False  # the low-level per-conformer ticker is redundant with LazyMIL's own step progress
    return calculator.run(conf_list)


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


def target_fallback(y: Iterable[Any], task_type: str) -> Any:
    """Fallback prediction for molecules that can't be processed at inference time.

    Args:
        y (Iterable[Any]): Training targets.
        task_type (str): ``"continuous"`` or ``"binary"``.

    Returns:
        Any: The training target mean (continuous) or most frequent class (binary).
    """

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
    estimator_instance: Any,
    hopt: bool = True,
    seed: int = 42,
    accelerator: str = "cpu",
) -> tuple[list[Any], list[Any], list[Any]]:
    """Fit one estimator, refit on train+val, and predict on train/val/test - all in one call.

    There's nothing left over to persist afterward: the estimator and its scaler are local to this
    call, since predictions for every split are produced right here instead of via a later predict().

    Args:
        x_train (list[np.ndarray]): Training descriptor bags.
        x_val (list[np.ndarray]): Validation descriptor bags.
        x_test (list[np.ndarray]): Descriptor bags to predict on; may be empty.
        y_train (array-like): Training targets.
        y_val (array-like): Validation targets.
        estimator_instance: A MIL estimator implementing ``fit``/``predict``, optionally ``hopt``.
        hopt (bool): Whether to run ``estimator_instance.hopt`` before fitting, if supported.
        seed (int): Random seed passed to the estimator's hyperparameter search.
        accelerator (str): ``"cpu"``/``"gpu"``, forced into the hopt search grid so it can't be
            silently overridden by ``HYPERPARAMETERS``'s own fixed value.

    Returns:
        tuple: ``(pred_train, pred_val, pred_test)``. ``pred_train``/``pred_val`` come from the
        train-only fit; ``pred_test`` comes from the final train+val refit (empty if ``x_test`` is).
    """

    # 1. Scale train/val descriptors
    x_train_scaled, x_val_scaled = scale_descriptors(x_train, x_val)

    # 2. Optimize hyperparameters
    if hopt and hasattr(estimator_instance, "hopt"):
        param_grid = {**HYPERPARAMETERS, "random_seed": seed, "accelerator": accelerator}
        estimator_instance.hopt(x_train_scaled, y_train, param_grid=param_grid, verbose=False)

    # 3. Train on train split only (not final training yet)
    estimator_instance.fit(x_train_scaled, y_train)
    pred_train = list(estimator_instance.predict(x_train_scaled))
    pred_val = list(estimator_instance.predict(x_val_scaled))

    # 4. Retrain on the full train+val set and predict on test with that same fit.
    x_full = x_train + x_val
    y_full = np.hstack((y_train, y_val))
    x_full_scaled, x_test_scaled = scale_descriptors(x_full, x_test)
    estimator_instance.fit(x_full_scaled, y_full)
    pred_test = list(estimator_instance.predict(x_test_scaled)) if x_test else []

    return pred_train, pred_val, pred_test


class LazyMIL:
    """Train every built-in descriptor/estimator combination and predict on train/val/test in one pass.

    Everything happens inside a single call to :meth:`run` - there's no serialization and no separate
    predict() step, so nothing about a trained model is kept around afterward.
    """

    def __init__(
        self,
        task: str,
        hopt: bool = True,
        num_conf: int = 10,
        num_cpu: int = os.cpu_count() or 1,
        output_folder: str | None = None,
        verbose: bool = True,
        seed: int = 42,
        accelerator: str = "cpu",
    ) -> None:
        """Store settings and (re)create the output folder.

        Args:
            task (str): ``"continuous"`` or ``"binary"`` - selects REGRESSORS or CLASSIFIERS. Always
                supplied by the caller (:class:`~qsarmil.modelling.meta.MultiConformerEstimator`
                already knows its own task), so this isn't validated here.
            hopt (bool): Whether to hyperparameter-tune each estimator before fitting, if supported.
            num_conf (int): Number of conformers to generate per molecule.
            num_cpu (int): Number of CPU threads to use for conformer generation.
            output_folder (str, optional): Output directory; a fresh temp dir is created and wiped if omitted/exists.
            verbose (bool): Whether to print per-step progress.
            seed (int): Random seed for embedding and hyperparameter search.
            accelerator (str): ``"cpu"`` or ``"gpu"``, passed straight through to every estimator
                (construction and hyperparameter search alike).
        """
        self.task = task
        self.ESTIMATORS = REGRESSORS if task == "continuous" else CLASSIFIERS
        self.hopt = hopt
        self.num_conf = num_conf
        self.output_folder: str = output_folder or tempfile.mkdtemp(prefix="qsarmil_")
        self.num_cpu = num_cpu
        self.verbose = verbose
        self.seed = seed
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
        """Train every descriptor/estimator combination and predict on train/val/test in one pass.

        Pipeline: parse SMILES, generate conformers, and calculate descriptors for train+val+test
        together (one pass per descriptor type, for efficiency), then train every descriptor/estimator
        combination and predict on all three splits at once.

        Args:
            smiles_train (Sequence[str]): Training SMILES strings.
            y_train (Sequence[Any]): Training targets, same length/order as ``smiles_train``.
            smiles_val (Sequence[str]): Validation SMILES strings.
            y_val (Sequence[Any]): Validation targets, same length/order as ``smiles_val``.
            smiles_test (Sequence[str]): SMILES strings to predict on. Every row is kept in the
                output; molecules that fail parsing/conformer generation are imputed with the
                training set fallback rather than dropped.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ``(result_df_train, result_df_val,
            result_df_test)`` - each a ``SMILES`` column (train/val also get ``Y_TRUE``) plus one
            prediction column per descriptor/estimator combination. Also written to
            ``train.csv``/``val.csv``/``test.csv`` in ``self.output_folder``.
        """

        smi_train, y_train = list(smiles_train), list(y_train)
        smi_val, y_val = list(smiles_val), list(y_val)
        smi_test = list(smiles_test)

        # 1. Parse SMILES.
        if self.verbose:
            print("Step-1. SMILES parsing")
        smi_all = smi_train + smi_val + smi_test
        mols_all = parse_smiles(smi_all, verbose=self.verbose)

        # 2. Generate conformers.
        if self.verbose:
            print("Step-2. Conformer generation")
        conf_all = generate_conformers(
            mols_all, num_conf=self.num_conf, num_cpu=self.num_cpu, verbose=self.verbose, seed=self.seed
        )
        n_train, n_val = len(smi_train), len(smi_val)
        conf_train = conf_all[:n_train]
        conf_val = conf_all[n_train : n_train + n_val]
        conf_test = conf_all[n_train + n_val :]

        # Train/val: drop molecules that failed - training needs clean data.
        ok_train = [i for i, c in enumerate(conf_train) if isinstance(c, list)]
        smi_train = [smi_train[i] for i in ok_train]
        y_train = [y_train[i] for i in ok_train]
        conf_train = [conf_train[i] for i in ok_train]

        ok_val = [i for i, c in enumerate(conf_val) if isinstance(c, list)]
        smi_val = [smi_val[i] for i in ok_val]
        y_val = [y_val[i] for i in ok_val]
        conf_val = [conf_val[i] for i in ok_val]

        # Test: keep every row - the output needs one prediction per input SMILES. Molecules that
        # failed get the training set fallback instead of a real prediction, below.
        train_fallback = target_fallback(y_train, self.task)
        ok_test = [i for i, c in enumerate(conf_test) if isinstance(c, list)]
        smi_test_valid = [smi_test[i] for i in ok_test]
        conf_test_valid = [conf_test[i] for i in ok_test]

        n_failed_test = len(smi_test) - len(smi_test_valid)
        if n_failed_test and self.verbose:
            print(
                f"{n_failed_test} test molecule(s) could not be processed and will be predicted "
                "using the training set fallback value instead."
            )

        result_df_train = pd.DataFrame({"SMILES": smi_train, "Y_TRUE": y_train})
        result_df_val = pd.DataFrame({"SMILES": smi_val, "Y_TRUE": y_val})
        result_df_test = pd.DataFrame({"SMILES": smi_test})

        # 3. Calculate descriptors for train+val+test together, once per descriptor type.
        if self.verbose:
            print("Step-3. Descriptor calculation")

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

        # 4. Train every descriptor/estimator combination and predict on train/val/test.
        if self.verbose:
            print("Step-4. Model training")

        total_models = len(DESCRIPTORS) * len(self.ESTIMATORS)
        current_model = 0

        for desc_name, (x_train, x_val, x_test) in ready_descriptors.items():
            for est_name, factory in self.ESTIMATORS.items():
                estimator = factory(accelerator=self.accelerator)
                model_name = f"{desc_name}|{est_name}"
                current_model += 1

                with OutputSuppressor():
                    pred_train, pred_val, pred_test = train_estimator(
                        x_train, x_val, x_test, y_train, y_val, estimator, self.hopt, seed=self.seed,
                        accelerator=self.accelerator,
                    )

                preds_by_smi = dict(zip(smi_test_valid, pred_test))

                result_df_train[model_name] = pred_train
                result_df_val[model_name] = pred_val
                result_df_test[model_name] = [preds_by_smi.get(smi, train_fallback) for smi in smi_test]

                result_df_train.to_csv(os.path.join(self.output_folder, "train.csv"), index=False)
                result_df_val.to_csv(os.path.join(self.output_folder, "val.csv"), index=False)
                result_df_test.to_csv(os.path.join(self.output_folder, "test.csv"), index=False)

                if self.verbose:
                    print(f"[{current_model}/{total_models}] {model_name}")

        return result_df_train, result_df_val, result_df_test
