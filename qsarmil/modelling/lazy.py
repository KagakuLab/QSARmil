from __future__ import annotations
# ruff: noqa: I001

from collections.abc import Callable, Iterable
import os
import shutil
import tempfile
import time
from importlib import import_module
from typing import Any, Sequence

import numpy as np
import pandas as pd
import psutil

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
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from qsarmil.conformer.rdkit import RDKitConformerGenerator
from qsarmil.descriptor.wrapper import DescriptorWrapper

from qsarmil.utils.logging import FailedConformer, FailedMolecule, OutputSuppressor, print_step_header




RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
_VALID_ACCELERATORS = ("cpu", "gpu")


def _validate_accelerator(accelerator: str) -> str:
    """Require an explicit 'cpu' or 'gpu' choice - no 'auto' fallback to whatever hardware happens to be present."""
    if accelerator not in _VALID_ACCELERATORS:
        raise ValueError(f"accelerator must be one of {_VALID_ACCELERATORS!r}, got {accelerator!r}")
    return accelerator


# ==========================================================
# Configuration
# ==========================================================
DESCRIPTORS = {
    "RDKitGEOM": DescriptorWrapper(RDKitGEOM()),
    "RDKitAUTOCORR": DescriptorWrapper(RDKitAUTOCORR()),
    "RDKitRDF": DescriptorWrapper(RDKitRDF()),
    "RDKitMORSE": DescriptorWrapper(RDKitMORSE()),
    "RDKitWHIM": DescriptorWrapper(RDKitWHIM()),
    "MolFeatUSRD": DescriptorWrapper(USRDescriptors()),
    "MolFeatElectroShape": DescriptorWrapper(ElectroShapeDescriptors()),
    "RDKitGETAWAY": DescriptorWrapper(RDKitGETAWAY()),
    "MolFeatPmapper": DescriptorWrapper(Pharmacophore3D(factory="pmapper")),
}

REGRESSORS = {
    # mil wrappers
    "MeanInstanceWrapperMLPNetworkRegressor": InstanceWrapperMLPNetworkRegressor(pool="mean"),
    "MeanBagWrapperMLPNetworkRegressor": BagWrapperMLPNetworkRegressor(pool="mean"),
    # mil networks
    "MeanBagNetworkRegressor": BagNetworkRegressor(pool="mean"),
    "MeanInstanceNetworkRegressor": InstanceNetworkRegressor(pool="mean"),
    "AdditiveAttentionNetworkRegressor": AdditiveAttentionNetworkRegressor(),
    "SelfAttentionNetworkRegressor": SelfAttentionNetworkRegressor(),
    "HopfieldAttentionNetworkRegressor": HopfieldAttentionNetworkRegressor(),
    "DynamicPoolingNetworkRegressor": DynamicPoolingNetworkRegressor(),
}

CLASSIFIERS =  {
    # mil wrappers
    "MeanInstanceWrapperMLPNetworkClassifier": InstanceWrapperMLPNetworkClassifier(pool="mean"),
    "MeanBagWrapperMLPNetworkClassifier": BagWrapperMLPNetworkClassifier(pool="mean"),
    # mil networks
    "MeanBagNetworkClassifier": BagNetworkClassifier(pool="mean"),
    "MeanInstanceNetworkClassifier": InstanceNetworkClassifier(pool="mean"),
    "AdditiveAttentionNetworkClassifier": AdditiveAttentionNetworkClassifier(),
    "SelfAttentionNetworkClassifier": SelfAttentionNetworkClassifier(),
    "HopfieldAttentionNetworkClassifier": HopfieldAttentionNetworkClassifier(),
    "DynamicPoolingNetworkClassifier": DynamicPoolingNetworkClassifier(),
}

DEFAULT_PARAM_GRID = {
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
# Utility Functions
# ==========================================================

def gen_conformers(
    smi_list: Iterable[str], num_conf: int = 10, num_cpu: int = os.cpu_count() or 1, verbose: bool = False, seed: int = 42
) -> list[list[Any] | FailedMolecule | FailedConformer]:
    """Generate conformers per SMILES; unparseable/failed-embedding molecules become Failed* sentinels, not raises."""
    mol_list = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = FailedMolecule(smi)
        mol_list.append(mol)
    conf_gen = RDKitConformerGenerator(num_conf=num_conf, num_cpu=num_cpu, verbose=verbose, seed=seed)
    conf_list = conf_gen.run(mol_list)
    return conf_list

def report_smiles_parsing(smiles: list[str], confs: list[Any], verbose: bool = True) -> set[int]:
    """Report Step 1 (SMILES parsing) outcomes and return the failed row indices.

    Args:
        smiles (list[str]): SMILES strings, in their original order.
        confs (list[Any]): Per-SMILES :func:`gen_conformers` output, same order.
        verbose (bool): Whether to print the report.

    Returns:
        set[int]: Indices into ``smiles`` whose SMILES failed to parse.
    """

    failed_idx = {i for i, c in enumerate(confs) if isinstance(c, FailedMolecule)}

    if verbose:
        print_step_header(1, "SMILES parsing")
        n_total = len(smiles)
        n_ok = n_total - len(failed_idx)
        print(f"> For {n_ok} of {n_total} molecules, SMILES were parsed correctly.")
        if failed_idx:
            print(f"> For {len(failed_idx)} molecules, SMILES could not be parsed and were removed from training:")
            for i in sorted(failed_idx):
                print(f"       - Row {i}:  {smiles[i]}")

    return failed_idx


def report_conformer_generation(
    smiles: list[str], confs: list[Any], already_failed: set[int], verbose: bool = True
) -> set[int]:
    """Report Step 2 (conformer generation) outcomes and return the failed row indices.

    Args:
        smiles (list[str]): SMILES strings, in their original order.
        confs (list[Any]): Per-SMILES :func:`gen_conformers` output, same order.
        already_failed (set[int]): Indices already dropped in Step 1, excluded from the "considered" count.
        verbose (bool): Whether to print the report.

    Returns:
        set[int]: Indices into ``smiles`` whose 3D embedding failed.
    """

    failed_idx = {i for i, c in enumerate(confs) if isinstance(c, FailedConformer)}
    ok_idx = [i for i, c in enumerate(confs) if isinstance(c, list)]

    if verbose:
        print_step_header(2, "Conformer generation")
        n_considered = len(smiles) - len(already_failed)
        print(f"> For {len(ok_idx)} of {n_considered} molecules, conformers were generated successfully.")
        if ok_idx:
            counts = [len(confs[i]) for i in ok_idx]
            avg, mn, mx = np.mean(counts), min(counts), max(counts)
            print(f"> Average num conf: {avg:.1f} | min num conf: {mn} | max num conf: {mx}")
        if failed_idx:
            print(
                f"> For {len(failed_idx)} molecules, conformer generation failed, "
                "and they were removed from training:"
            )
            for i in sorted(failed_idx):
                print(f"       - Row {i}:  {smiles[i]}")

    return failed_idx

def _subset(items: Sequence[Any], idx: Sequence[int]) -> list[Any]:
    """Pick out ``items[i]`` for each ``i`` in ``idx``, preserving ``idx``'s order."""

    return [items[i] for i in idx]

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


def _print_progress_item(index: int, total: int, label: str, elapsed_min: float, mem_gb: float) -> None:
    """Print a ``[i/n] label`` line followed by an indented, hanging-aligned timing/memory line."""

    prefix = f"[{index}/{total}] "
    print(f"{prefix}{label}")
    print(f"{' ' * len(prefix)}> Finished in {elapsed_min:.2f} min | Memory usage: {mem_gb:.3f} G")


def calc_descriptors(conf_list: list[list[Any]], calculator: DescriptorWrapper) -> list[np.ndarray]:
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

# ==========================================================
# ModelBuilder Class
# ==========================================================
def build_model(
    x_train: list[np.ndarray],
    x_val: list[np.ndarray],
    y_train: Iterable[Any],
    y_val: Iterable[Any],
    estimator_instance: Any,
    hopt: bool = True,
    seed: int = 42,
    accelerator: str = "auto",
) -> tuple[list[Any], list[Any], Any, BagMinMaxScaler]:
    """Fit one estimator and refit on train+val, returning predictions and the artifacts to persist.

    Args:
        x_train (list[np.ndarray]): Training descriptor bags.
        x_val (list[np.ndarray]): Validation descriptor bags.
        y_train (array-like): Training targets.
        y_val (array-like): Validation targets.
        estimator_instance: A MIL estimator implementing ``fit``/``predict``, optionally ``hopt``.
        hopt (bool): Whether to run ``estimator_instance.hopt`` before fitting, if supported.
        seed (int): Random seed passed to the estimator's hyperparameter search.
        accelerator (str): ``"auto"``/``"cpu"``/``"gpu"``, forced into the hopt search grid so it
            can't be silently overridden by ``DEFAULT_PARAM_GRID``'s own fixed value.

    Returns:
        tuple: ``(pred_train, pred_val, fitted_estimator, fitted_scaler)`` from the final train+val refit.
    """

    # 1. Scale train/val descriptors
    x_train_scaled, x_val_scaled = scale_descriptors(x_train, x_val)

    # 2. Optimize hyperparameters
    if hopt and hasattr(estimator_instance, "hopt"):
        param_grid = {**DEFAULT_PARAM_GRID, "random_seed": seed, "accelerator": accelerator}
        estimator_instance.hopt(x_train_scaled, y_train, param_grid=param_grid, verbose=False)

    # 3. Train on train split only (not final training yet)
    estimator_instance.fit(x_train_scaled, y_train)
    pred_train = list(estimator_instance.predict(x_train_scaled))
    pred_val = list(estimator_instance.predict(x_val_scaled))

    # 4. Retrain model on full (train + val) - this is the artifact that
    #    gets persisted for later inference.
    x_full, y_full = x_train + x_val, np.hstack((y_train, y_val))
    scaler_full = BagMinMaxScaler()
    scaler_full.fit(x_full)
    x_full_scaled = scaler_full.transform(x_full)
    estimator_instance.fit(x_full_scaled, y_full)

    return pred_train, pred_val, estimator_instance, scaler_full


class LazyMIL:
    """Train every built-in descriptor/estimator combination on one dataset; use predict() for new data.

    Trains, then predicts, within the same process/session - there's no serialization; all fitted models
    and descriptor calculators are kept in memory for the lifetime of this object.
    """

    def __init__(
        self,
        hopt: bool = True,
        num_conf: int = 10,
        num_cpu: int = os.cpu_count() or 1,
        output_folder: str | None = None,
        verbose: bool = True,
        seed: int = 42,
        val_size: float = 0.2,
        task: str | None = None,
        accelerator: str = "cpu",
    ) -> None:
        """Store settings and (re)create the output folder.

        Args:
            hopt (bool): Whether to hyperparameter-tune each estimator before fitting, if supported.
            num_conf (int): Number of conformers to generate per molecule.
            num_cpu (int): Number of CPU threads to use for conformer generation.
            output_folder (str, optional): Output directory; a fresh temp dir is created and wiped if omitted/exists.
            verbose (bool): Whether to print per-model progress and memory usage.
            seed (int): Random seed for embedding, validation, the train/val split, and hyperparameter search.
            val_size (float): Fraction of the data held out as a random validation split inside :meth:`run`.
            task (str, optional): ``"continuous"`` or ``"binary"`` to force the task, skipping auto-detection.
            accelerator (str): ``"cpu"`` or ``"gpu"`` - an explicit choice, never auto-detected. Used for
                training in :meth:`run`; :meth:`predict` can override it per call.
        """
        self.hopt = hopt
        self.num_conf = num_conf
        self.output_folder: str = output_folder or tempfile.mkdtemp(prefix="qsarmil_")
        self.num_cpu = num_cpu
        self.verbose = verbose
        self.seed = seed
        self.val_size = val_size
        self.task = task
        self.accelerator = _validate_accelerator(accelerator)

        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)

        # Populated by run(); reused by predict() within the same process -
        # no persistence to disk, so nothing here survives past this object.
        self._trained_models: dict[str, dict[str, Any]] = {}
        self._fitted_descriptors: dict[str, DescriptorWrapper] = {}
        self._task_type: str | None = None
        self._train_fallback: Any = None

    @property
    def is_trained(self) -> bool:
        """Whether :meth:`run` has produced at least one trained model."""

        return bool(self._trained_models)

    def run(self, smiles: Sequence[str], y: Sequence[Any]) -> None:
        """Train every descriptor/estimator combination and write predictions to CSV.

        Args:
            smiles (Sequence[str]): SMILES strings.
            y (Sequence[Any]): Target property value for each SMILES, same length and order as ``smiles``.

        Returns:
            None. Writes ``train.csv``/``val.csv`` to ``self.output_folder``; use :meth:`predict` for new data.
        """

        smi_all, y_all = list(smiles), list(y)

        # 1 & 2. Parse SMILES and generate conformers once for the whole
        #    dataset, then report+drop molecules that fail either step.
        conf_all = gen_conformers(
            smi_all, num_conf=self.num_conf, num_cpu=self.num_cpu, verbose=False, seed=self.seed
        )
        parse_failed = report_smiles_parsing(smi_all, conf_all, self.verbose)
        conf_failed = report_conformer_generation(smi_all, conf_all, parse_failed, self.verbose)

        keep_idx = [i for i in range(len(smi_all)) if i not in parse_failed and i not in conf_failed]
        smi_all, y_all, conf_all = _subset(smi_all, keep_idx), _subset(y_all, keep_idx), _subset(conf_all, keep_idx)

        # 3. Get a task type
        task_type = self.task if self.task is not None else type_of_target(y_all)
        if task_type == "continuous":
            estimators_source = REGRESSORS
        elif task_type == "binary":
            estimators_source = CLASSIFIERS
        else:
            raise ValueError(
                f"Task type '{task_type}' not supported (only 'continuous' and 'binary' targets are supported)."
            )
        self._task_type = task_type

        # 4. Cache the fallback prediction (mean / most frequent class over
        #    the whole dataset) used by `predict` for molecules it can't
        #    process later.
        self._train_fallback = target_fallback(y_all, task_type)

        # 5. Random train/validation split.
        idx_train, idx_val = train_test_split(
            range(len(smi_all)), test_size=self.val_size, random_state=self.seed
        )
        smi_train, y_train, conf_train = _subset(smi_all, idx_train), _subset(y_all, idx_train), _subset(
            conf_all, idx_train
        )
        smi_val, y_val, conf_val = _subset(smi_all, idx_val), _subset(y_all, idx_val), _subset(conf_all, idx_val)

        result_df_train = pd.DataFrame({"SMILES": smi_train, "Y_TRUE": y_train})
        result_df_val = pd.DataFrame({"SMILES": smi_val, "Y_TRUE": y_val})

        # 6. Calculate descriptors for every descriptor set.
        if self.verbose:
            print_step_header(3, "Descriptor calculation")

        per_descriptor: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
        self._fitted_descriptors = {}
        for d_i, (desc_name, desc_source) in enumerate(DESCRIPTORS.items(), start=1):
            desc_calc = desc_source()

            start = time.time()
            x_train = calc_descriptors(conf_train, desc_calc)
            x_val = calc_descriptors(conf_val, desc_calc)
            elapsed_min = (time.time() - start) / 60
            mem_gb = psutil.Process().memory_info().rss / (1024**3)

            per_descriptor[desc_name] = (x_train, x_val)
            self._fitted_descriptors[desc_name] = desc_calc  # remembers its own learned column-drop decision

            if self.verbose:
                _print_progress_item(d_i, len(DESCRIPTORS), f"{desc_name}:", elapsed_min, mem_gb)

        # 7. Train every descriptor/estimator combination.
        if self.verbose:
            print_step_header(4, "Individual model building")

        total_models = len(DESCRIPTORS) * len(estimators_source)
        current_model = 0
        self._trained_models = {}

        for desc_name, (x_train, x_val) in per_descriptor.items():
            for est_name, factory in estimators_source.items():
                estimator = factory(accelerator=self.accelerator)

                model_name = f"{desc_name}|{est_name}"
                current_model += 1

                start = time.time()
                with OutputSuppressor():
                    pred_train, pred_val, fitted_estimator, fitted_scaler = build_model(
                        x_train, x_val, y_train, y_val, estimator, self.hopt, seed=self.seed,
                        accelerator=self.accelerator,
                    )
                elapsed_min = (time.time() - start) / 60
                mem_gb = psutil.Process().memory_info().rss / (1024**3)

                self._trained_models[model_name] = {
                    "descriptor": desc_name,
                    "estimator": fitted_estimator,
                    "scaler": fitted_scaler,
                }

                # Write predictions
                result_df_train[model_name] = pred_train
                result_df_train.to_csv(os.path.join(self.output_folder, "train.csv"), index=False)

                result_df_val[model_name] = pred_val
                result_df_val.to_csv(os.path.join(self.output_folder, "val.csv"), index=False)

                if self.verbose:
                    _print_progress_item(current_model, total_models, model_name, elapsed_min, mem_gb)

    def predict(self, smiles: Sequence[str], save: bool = False) -> pd.DataFrame:
        """Run inference from in-memory fitted models, imputing molecules that fail with :attr:`_train_fallback`.

        Args:
            smiles (Sequence[str]): SMILES strings to predict on.
            save (bool): Whether to also write the result to ``test.csv`` in ``self.output_folder``.

        Returns:
            pd.DataFrame: A ``SMILES`` column plus one prediction column per trained descriptor/estimator combo.
        """

        if not self.is_trained:
            raise RuntimeError("LazyMIL is not trained. Call `run` first.")

        smi_test = list(smiles)
        result_df_test = pd.DataFrame({"SMILES": smi_test})

        confs = gen_conformers(smi_test, num_conf=self.num_conf, num_cpu=self.num_cpu, verbose=False, seed=self.seed)
        failed_smiles = {
            smi for smi, c in zip(smi_test, confs) if isinstance(c, (FailedMolecule, FailedConformer))
        }

        if failed_smiles and self.verbose:
            print(
                f"\n{len(failed_smiles)} molecule(s) could not be processed and will be "
                "predicted using the training set fallback value instead:"
            )
            for i, smi in enumerate(smi_test):
                if smi in failed_smiles:
                    print(f"  > Row {i}: {smi}")

        valid_smi = [smi for smi, c in zip(smi_test, confs) if smi not in failed_smiles]
        valid_confs = [c for smi, c in zip(smi_test, confs) if smi not in failed_smiles]

        descriptor_names = {model_state["descriptor"] for model_state in self._trained_models.values()}
        x_by_descriptor: dict[str, list[np.ndarray]] = {}
        if valid_confs:
            for desc_name in descriptor_names:
                desc_calc = self._fitted_descriptors[desc_name]
                x_by_descriptor[desc_name] = calc_descriptors(valid_confs, desc_calc)

        for model_name, model_state in self._trained_models.items():
            desc_name = model_state["descriptor"]

            preds_by_smi: dict[str, Any] = {}
            if valid_smi:
                x_test = x_by_descriptor[desc_name]
                x_test_scaled = model_state["scaler"].transform(x_test)
                with OutputSuppressor():
                    preds = model_state["estimator"].predict(x_test_scaled)
                preds_by_smi = dict(zip(valid_smi, preds))

            result_df_test[model_name] = [preds_by_smi.get(smi, self._train_fallback) for smi in smi_test]

        if save:
            result_df_test.to_csv(os.path.join(self.output_folder, "test.csv"), index=False)
        return result_df_test
