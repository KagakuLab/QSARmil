from __future__ import annotations
# ruff: noqa: I001

from collections.abc import Callable, Iterable
import os
import pickle
import shutil
import tempfile
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import psutil

from milearn.preprocessing import BagMinMaxScaler
from milearn.wrapper import BagWrapper
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
def model_factory(module_name: str, class_name: str, /, *args: Any, **kwargs: Any) -> Any:
    is_milearn = module_name.startswith("milearn.")

    def build(accelerator: str | None = None) -> Any:
        cls = getattr(import_module(module_name), class_name)
        call_kwargs = dict(kwargs)
        if is_milearn and accelerator is not None:
            call_kwargs["accelerator"] = accelerator
        # wrap non-MIL models
        if not is_milearn:
            return BagWrapper(cls(*args, **call_kwargs))
        return cls(*args, **call_kwargs)

    return build

def descriptor_factory(
    module_name: str, class_name: str, /, *args: Any, **kwargs: Any
) -> Callable[[], DescriptorWrapper]:
    def build() -> DescriptorWrapper:
        cls = getattr(import_module(module_name), class_name)
        return DescriptorWrapper(cls(*args, **kwargs))

    return build

def _DESCRIPTORS() -> dict[str, Callable[[], DescriptorWrapper]]:


    return {
        "RDKitGEOM": descriptor_factory("qsarmil.descriptor.rdkit", "RDKitGEOM"),
        "RDKitAUTOCORR": descriptor_factory("qsarmil.descriptor.rdkit", "RDKitAUTOCORR"),
        "RDKitRDF": descriptor_factory("qsarmil.descriptor.rdkit", "RDKitRDF"),
        "RDKitMORSE": descriptor_factory("qsarmil.descriptor.rdkit", "RDKitMORSE"),
        "RDKitWHIM": descriptor_factory("qsarmil.descriptor.rdkit", "RDKitWHIM"),
        "MolFeatUSRD": descriptor_factory("molfeat.calc", "USRDescriptors"),
        "MolFeatElectroShape": descriptor_factory("molfeat.calc", "ElectroShapeDescriptors"),
        "RDKitGETAWAY": descriptor_factory("qsarmil.descriptor.rdkit", "RDKitGETAWAY"),
        "MolFeatPmapper": descriptor_factory("molfeat.calc", "Pharmacophore3D", factory="pmapper"),
    }


DESCRIPTORS = _DESCRIPTORS()

def _REGRESSORS() -> dict[str, Any]:
    return {
        # mil wrappers
        "MeanInstanceWrapperMLPNetworkRegressor": model_factory(
            "milearn.network.regressor", "InstanceWrapperMLPNetworkRegressor", pool="mean"
        ),
        "MeanBagWrapperMLPNetworkRegressor": model_factory(
            "milearn.network.regressor", "BagWrapperMLPNetworkRegressor", pool="mean"
        ),
        # mil networks
        "MeanBagNetworkRegressor": model_factory("milearn.network.regressor", "BagNetworkRegressor", pool="mean"),
        "MeanInstanceNetworkRegressor": model_factory(
            "milearn.network.regressor", "InstanceNetworkRegressor", pool="mean"
        ),
        "AdditiveAttentionNetworkRegressor": model_factory(
            "milearn.network.regressor", "AdditiveAttentionNetworkRegressor"
        ),
        "SelfAttentionNetworkRegressor": model_factory("milearn.network.regressor", "SelfAttentionNetworkRegressor"),
        "HopfieldAttentionNetworkRegressor": model_factory(
            "milearn.network.regressor", "HopfieldAttentionNetworkRegressor"
        ),
        "DynamicPoolingNetworkRegressor": model_factory("milearn.network.regressor", "DynamicPoolingNetworkRegressor"),
    }


def _CLASSIFIERS() -> dict[str, Any]:
    return {
        # mil wrappers
        "MeanInstanceWrapperMLPNetworkClassifier": model_factory(
            "milearn.network.classifier", "InstanceWrapperMLPNetworkClassifier", pool="mean"
        ),
        "MeanBagWrapperMLPNetworkClassifier": model_factory(
            "milearn.network.classifier", "BagWrapperMLPNetworkClassifier", pool="mean"
        ),
        # mil networks
        "MeanBagNetworkClassifier": model_factory("milearn.network.classifier", "BagNetworkClassifier", pool="mean"),
        "MeanInstanceNetworkClassifier": model_factory(
            "milearn.network.classifier", "InstanceNetworkClassifier", pool="mean"
        ),
        "AdditiveAttentionNetworkClassifier": model_factory(
            "milearn.network.classifier", "AdditiveAttentionNetworkClassifier"
        ),
        "SelfAttentionNetworkClassifier": model_factory(
            "milearn.network.classifier", "SelfAttentionNetworkClassifier"
        ),
        "HopfieldAttentionNetworkClassifier": model_factory(
            "milearn.network.classifier", "HopfieldAttentionNetworkClassifier"
        ),
        "DynamicPoolingNetworkClassifier": model_factory(
            "milearn.network.classifier", "DynamicPoolingNetworkClassifier"
        ),
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

# ==========================================================
# Utility Functions
# ==========================================================

def gen_conformers(
    smi_list: Iterable[str], num_conf: int = 10, num_cpu: int = 1, verbose: bool = False, seed: int = 42
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


def calc_descriptors(
    conf_list: list[list[Any]],
    calculator: DescriptorWrapper,
    verbose: bool = False,
    col_stats: dict[str, np.ndarray] | None = None,
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Compute and clean descriptor bags for a list of per-molecule conformer bags."""
    calculator.verbose = False  # the low-level per-conformer ticker is redundant with LazyMIL's own step progress
    return calculator.run(conf_list, verbose=verbose, col_stats=col_stats)

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
    """Train every built-in descriptor/estimator combination on one dataset; use predict() for new data."""

    def __init__(
        self,
        hopt: bool = True,
        num_conf: int = 10,
        num_cpu: int = 20,
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

        # Populated after ``run`` and reused by ``predict`` for inference-only
        # execution (descriptor generation + estimator.predict only).
        self._trained_models: dict[str, dict[str, Any]] = {}
        self._task_type: str | None = None
        self._train_fallback: Any = None

        # Descriptor cache: persistent storage for computed descriptors
        # DataFrame with columns: descriptor_name, SMILES, descriptor_vector
        self._descriptor_cache_path = os.path.join(self.output_folder, "descriptor_cache.pkl")
        self._descriptor_cache: pd.DataFrame = pd.DataFrame(
            columns=["descriptor_name", "SMILES", "descriptor_vector"]
        )

    @property
    def is_trained(self) -> bool:
        """Whether this instance has serialized-ready fitted model artifacts."""

        return bool(self._trained_models)

    def _ensure_estimator_predict_ready(self, estimator_instance: Any, accelerator: str) -> None:
        """Rebuild a milearn estimator's runtime trainer (on ``accelerator``) after unpickling."""

        module_name = estimator_instance.__class__.__module__
        if not module_name.startswith("milearn."):
            return

        if not hasattr(estimator_instance, "_trainer"):
            return

        if estimator_instance._trainer is not None:
            return

        import pytorch_lightning as pl

        hparams = estimator_instance.hparams
        estimator_instance._trainer = pl.Trainer(
            max_epochs=getattr(hparams, "max_epochs", 1),
            callbacks=[],
            logger=False,
            accelerator=accelerator,
            enable_model_summary=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            deterministic=True,
        )

    def _load_descriptor_cache(self) -> None:
        """Load descriptor cache from disk if it exists."""
        if os.path.exists(self._descriptor_cache_path):
            loaded_data = pd.read_pickle(self._descriptor_cache_path)
            if isinstance(loaded_data, pd.DataFrame):
                self._descriptor_cache = loaded_data
            else:
                raise ValueError("Descriptor cache file not in the right format.")
        else:
            self._descriptor_cache = pd.DataFrame(
                columns=["descriptor_name", "SMILES", "descriptor_vector"]
            )

    def _save_descriptor_cache(self) -> None:
        """Save descriptor cache to disk."""
        self._descriptor_cache.to_pickle(self._descriptor_cache_path)

    def _get_cached_descriptors(
        self, desc_name: str, smi_list: list[str]
    ) -> tuple[list[np.ndarray | None], list[str]]:
        """Retrieve cached descriptors for the given SMILES, returning found vectors and the still-uncached ones.

        Args:
            desc_name: Name of the descriptor type.
            smi_list: SMILES strings to look up.

        Returns:
            tuple: ``(found_vectors, uncached_smiles)``.
        """
        smi_mask = self._descriptor_cache["SMILES"].isin(smi_list)
        mask = (self._descriptor_cache["descriptor_name"] == desc_name) & (
            smi_mask
        )
        results = self._descriptor_cache.loc[mask, :]
        not_found = [x for x in smi_list if x not in results["SMILES"].values]
        return results["descriptor_vector"], not_found

    def _cache_descriptor(self, desc_name: str, smi: list[str], descriptor: list[np.ndarray]) -> None:
        """Store a descriptor in cache using DataFrame append."""
        new_row = pd.DataFrame({
            "descriptor_name": len(smi) * [desc_name],
            "SMILES": smi,
            "descriptor_vector": descriptor,
        })
        self._descriptor_cache = pd.concat([self._descriptor_cache, new_row], ignore_index=True)
        self._save_descriptor_cache()

    def run(self, smiles: Sequence[str], y: Sequence[Any]) -> None:
        """Train every descriptor/estimator combination and write predictions to CSV.

        Args:
            smiles (Sequence[str]): SMILES strings.
            y (Sequence[Any]): Target property value for each SMILES, same length and order as ``smiles``.

        Returns:
            None. Writes ``train.csv``/``val.csv`` to ``self.output_folder``; use :meth:`predict` for new data.
        """

        # Reset previous fitted artifacts for a fresh training run.
        self._trained_models = {}

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

        # 6. Calculate descriptors for every descriptor set, imputing val
        #    NaNs with train's own column means.
        if self.verbose:
            print_step_header(3, "Descriptor calculation")

        per_descriptor: dict[str, tuple[list[np.ndarray], list[np.ndarray], dict[str, np.ndarray]]] = {}
        for d_i, (desc_name, desc_source) in enumerate(DESCRIPTORS.items(), start=1):
            desc_calc = desc_source()

            start = time.time()
            x_train, col_stats = calc_descriptors(conf_train, desc_calc, verbose=self.verbose)
            x_val, _ = calc_descriptors(conf_val, desc_calc, verbose=False, col_stats=col_stats)
            elapsed_min = (time.time() - start) / 60
            mem_gb = psutil.Process().memory_info().rss / (1024**3)

            per_descriptor[desc_name] = (x_train, x_val, col_stats)

            if self.verbose:
                _print_progress_item(d_i, len(DESCRIPTORS), f"{desc_name}:", elapsed_min, mem_gb)

        # 7. Train every descriptor/estimator combination.
        if self.verbose:
            print_step_header(4, "Individual model building")

        total_models = len(DESCRIPTORS) * len(estimators_source)
        current_model = 0

        for desc_name, (x_train, x_val, col_stats) in per_descriptor.items():
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

                # Persist everything needed for inference-only execution.
                self._trained_models[model_name] = {
                    "descriptor": desc_name,
                    "estimator": fitted_estimator,
                    "scaler": fitted_scaler,
                    "col_stats": col_stats,
                }

                # Write predictions
                result_df_train[model_name] = pred_train
                result_df_train.to_csv(os.path.join(self.output_folder, "train.csv"), index=False)

                result_df_val[model_name] = pred_val
                result_df_val.to_csv(os.path.join(self.output_folder, "val.csv"), index=False)

                if self.verbose:
                    _print_progress_item(current_model, total_models, model_name, elapsed_min, mem_gb)

    def predict(self, smiles: Sequence[str], save: bool = False, accelerator: str | None = None) -> pd.DataFrame:
        """Run inference from persisted fitted models, imputing molecules that fail with :attr:`_train_fallback`.

        Args:
            smiles (Sequence[str]): SMILES strings to predict on.
            save (bool): Whether to also write the result to ``test.csv`` in ``self.output_folder``.
            accelerator (str, optional): ``"cpu"`` or ``"gpu"`` to run inference on, overriding
                :attr:`accelerator` for this call only - e.g. predict on CPU for a model trained on GPU.
                Defaults to :attr:`accelerator` (the value set at construction/training time).

        Returns:
            pd.DataFrame: A ``SMILES`` column plus one prediction column per trained descriptor/estimator combo.
        """

        if not self.is_trained:
            raise RuntimeError("LazyMIL is not trained. Call `run` or `load` first.")

        accelerator = _validate_accelerator(accelerator if accelerator is not None else self.accelerator)

        # Load existing descriptor cache
        self._load_descriptor_cache()

        smi_test = list(smiles)
        result_df_test = pd.DataFrame({"SMILES": smi_test})

        descriptor_stats: dict[str, dict[str, np.ndarray]] = {}
        for model_state in self._trained_models.values():
            descriptor_stats[model_state["descriptor"]] = model_state["col_stats"]

        # Figure out, once, which SMILES need fresh conformers for at least
        # one descriptor (conformer embeddability doesn't depend on the
        # descriptor type, so this is shared across the loop below).
        smiles_needing: dict[str, list[str]] = {}
        all_needed: list[str] = []
        seen: set[str] = set()
        for desc_name in descriptor_stats:
            if desc_name not in DESCRIPTORS:
                raise ValueError(
                    f"Descriptor '{desc_name}' was used during training but isn't available in current DESCRIPTORS."
                )
            _, needing = self._get_cached_descriptors(desc_name, smi_test)
            smiles_needing[desc_name] = needing
            for smi in needing:
                if smi not in seen:
                    seen.add(smi)
                    all_needed.append(smi)

        confs_by_smiles: dict[str, Any] = {}
        if all_needed:
            confs = gen_conformers(
                all_needed, num_conf=self.num_conf, num_cpu=self.num_cpu, verbose=False, seed=self.seed
            )
            confs_by_smiles = dict(zip(all_needed, confs))

        failed_smiles = {
            smi for smi, c in confs_by_smiles.items() if isinstance(c, (FailedMolecule, FailedConformer))
        }

        if failed_smiles and self.verbose:
            print(
                f"\n{len(failed_smiles)} molecule(s) could not be processed and will be "
                "predicted using the training set fallback value instead:"
            )
            for i, smi in enumerate(smi_test):
                if smi in failed_smiles:
                    print(f"  > Row {i}: {smi}")

        for desc_name, col_stats in descriptor_stats.items():
            needing = [smi for smi in smiles_needing[desc_name] if smi not in failed_smiles]

            if needing:
                desc_calc = DESCRIPTORS[desc_name]()
                confs_subset = [confs_by_smiles[smi] for smi in needing]
                calculated_test_descs, _ = calc_descriptors(confs_subset, desc_calc, verbose=False, col_stats=col_stats)
                self._cache_descriptor(desc_name, needing, calculated_test_descs)

        valid_smi = [smi for smi in smi_test if smi not in failed_smiles]

        for model_name, model_state in self._trained_models.items():
            desc_name = model_state["descriptor"]
            scaler = model_state["scaler"]
            estimator = model_state["estimator"]

            preds_by_smi: dict[str, Any] = {}
            if valid_smi:
                x_test, missing = self._get_cached_descriptors(desc_name, valid_smi)
                assert not missing
                x_test_scaled = scaler.transform(x_test)
                with OutputSuppressor():
                    self._ensure_estimator_predict_ready(estimator, accelerator)
                    preds = estimator.predict(x_test_scaled)
                preds_by_smi = dict(zip(valid_smi, preds))

            result_df_test[model_name] = [preds_by_smi.get(smi, self._train_fallback) for smi in smi_test]

        if save:
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
            "val_size": self.val_size,
            "task": self.task,
            "accelerator": self.accelerator,
            "task_type": self._task_type,
            "train_fallback": self._train_fallback,
            "trained_models": self._trained_models,
        }
        with model_path.open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, model_path: str | Path, output_folder: str | None = None, accelerator: str | None = None) -> LazyMIL:
        """Load a serialized LazyMIL artifact for inference-only use.

        Args:
            model_path (str | Path): Path to a file written by :meth:`save`.
            output_folder (str, optional): Output directory; a fresh temp dir is created if omitted.
            accelerator (str, optional): ``"cpu"`` or ``"gpu"`` to override the accelerator this model was
                trained with - e.g. load a GPU-trained model but run inference on CPU from now on. Defaults
                to whatever accelerator was used at training time.
        """

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
            val_size=state.get("val_size", 0.2),
            task=state.get("task"),
            accelerator=accelerator if accelerator is not None else state.get("accelerator", "cpu"),
        )
        model._task_type = state.get("task_type")
        model._train_fallback = state.get("train_fallback")
        model._trained_models = state["trained_models"]
        return model

