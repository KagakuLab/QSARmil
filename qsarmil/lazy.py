import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import psutil
from milearn.network.module.hopt import DEFAULT_PARAM_GRID

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

# preprocessing
from milearn.preprocessing import BagMinMaxScaler
from milearn.wrapper import BagWrapper, InstanceWrapper
from molfeat.calc import ElectroShapeDescriptors, Pharmacophore3D, USRDescriptors
# descriptors
from rdkit import Chem, RDLogger
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR
from xgboost import XGBClassifier, XGBRegressor

from qsarmil.conformer.rdkit import RDKitConformerGenerator
from qsarmil.descriptor.rdkit import RDKitAUTOCORR, RDKitGEOM, RDKitGETAWAY, RDKitMORSE, RDKitRDF, RDKitWHIM
from qsarmil.descriptor.wrapper import DescriptorWrapper

from .utils.logging import OutputSuppressor

RDLogger.DisableLog("rdApp.*")

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
    # classic wrappers
    "MeanBagWrapperRidgeRegressor": BagWrapper(Ridge(), pool="mean"),
    "MeanBagWrapperLinearSVRRegressor": BagWrapper(LinearSVR(), pool="mean"),
    "MeanBagWrapperXGBRegressor": BagWrapper(XGBRegressor(), pool="mean"),
    "MeanBagWrapperMLPRegressor": BagWrapper(MLPRegressor(), pool="mean"),
    # classic wrappers
    "MeanInstanceWrapperRidgeRegressor": InstanceWrapper(Ridge(), pool="mean"),
    "MeanInstanceWrapperLinearSVRRegressor": InstanceWrapper(LinearSVR(), pool="mean"),
    "MeanInstanceWrapperXGBRegressor": InstanceWrapper(XGBRegressor(), pool="mean"),
    "MeanInstanceWrapperMLPRegressor": InstanceWrapper(MLPRegressor(), pool="mean"),
    # mil networks
    "MeanBagNetworkRegressor": BagNetworkRegressor(pool="mean"),
    "MeanInstanceNetworkRegressor": InstanceNetworkRegressor(pool="mean"),
    "AdditiveAttentionNetworkRegressor": AdditiveAttentionNetworkRegressor(),
    "SelfAttentionNetworkRegressor": SelfAttentionNetworkRegressor(),
    "HopfieldAttentionNetworkRegressor": HopfieldAttentionNetworkRegressor(),
    "DynamicPoolingNetworkRegressor": DynamicPoolingNetworkRegressor(),
}

CLASSIFIERS = {
    # classic wrappers
    "MeanBagWrapperRidgeClassifier": BagWrapper(RidgeClassifier(), pool="mean"),
    "MeanBagWrapperLinearSVCClassifier": BagWrapper(LinearSVC(), pool="mean"),
    "MeanBagWrapperXGBClassifier": BagWrapper(XGBClassifier(), pool="mean"),
    "MeanBagWrapperMLPClassifier": BagWrapper(MLPClassifier(), pool="mean"),
    # classic wrappers
    "MeanInstanceWrapperRidgeClassifier": InstanceWrapper(RidgeClassifier(), pool="mean"),
    "MeanInstanceWrapperLinearSVCClassifier": InstanceWrapper(LinearSVC(), pool="mean"),
    "MeanInstanceWrapperXGBClassifier": InstanceWrapper(XGBClassifier(), pool="mean"),
    "MeanInstanceWrapperMLPClassifier": InstanceWrapper(MLPClassifier(), pool="mean"),
    # mil networks
    "MeanBagNetworkClassifier": BagNetworkClassifier(pool="mean"),
    "MeanInstanceNetworkClassifier": InstanceNetworkClassifier(pool="mean"),
    "AdditiveAttentionNetworkClassifier": AdditiveAttentionNetworkClassifier(),
    "SelfAttentionNetworkClassifier": SelfAttentionNetworkClassifier(),
    "HopfieldAttentionNetworkClassifier": HopfieldAttentionNetworkClassifier(),
    "DynamicPoolingNetworkClassifier": DynamicPoolingNetworkClassifier(),
}


# ==========================================================
# Utility Functions
# ==========================================================
def _worker(func, args, kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return {"error": repr(e)}


def run_in_subprocess(func, *args, **kwargs):
    with ProcessPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_worker, func, args, kwargs)
        result = future.result()

    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])

    return result


def gen_conformers(smi_list, num_conf=10, num_cpu=1, verbose=False):
    """Generate conformers for a list of SMILES strings using
    RDKitConformerGenerator."""
    mol_list = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        mol_list.append(mol)
    conf_gen = RDKitConformerGenerator(num_conf=num_conf, num_cpu=num_cpu, verbose=verbose)
    conf_list = conf_gen.run(mol_list)
    return conf_list


def clean_descriptors(bags):
    """Replace NaN values in each bag's instances with the column means
    computed across all instances."""

    # Concatenate all instances from all bags into one 2D array
    all_instances = np.vstack(bags)

    # Compute column means ignoring NaNs
    col_means = np.nanmean(all_instances, axis=0)

    # Replace NaNs in each bag with the corresponding column mean
    cleaned_bags = []
    for bag in bags:
        bag = np.array(bag, dtype=float)  # Ensure float for NaN support
        idx = np.where(np.isnan(bag))
        bag[idx] = np.take(col_means, idx[1])
        cleaned_bags.append(bag)

    return cleaned_bags


def calc_descriptors(conf_list, calculator, verbose=False):
    calculator.verbose = verbose
    x = calculator.run(conf_list)
    x = clean_descriptors(x)
    return x


def scale_descriptors(x_train, x_test):
    scaler = BagMinMaxScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)


# ==========================================================
# ModelBuilder Class
# ==========================================================
def build_model(x_train, x_val, x_test, y_train, y_val, y_test, estimator_instance, hopt=True):

    # 1. Scale train/val descriptors
    x_train_scaled, x_val_scaled = scale_descriptors(x_train, x_val)

    # 2. Optimize hyperparameters
    if hopt and hasattr(estimator_instance, "hopt"):
        estimator_instance.hopt(x_train_scaled, y_train, param_grid=DEFAULT_PARAM_GRID, verbose=False)

    # 4. Train on train split only (not final training yet)
    estimator_instance.fit(x_train_scaled, y_train)
    pred_train = list(estimator_instance.predict(x_train_scaled))
    pred_val = list(estimator_instance.predict(x_val_scaled))

    # 5. Retrain model on full (train + val)
    x_full, y_full = x_train + x_val, np.hstack((y_train, y_val))
    x_full_scaled, x_test_scaled = scale_descriptors(x_full, x_test)
    estimator_instance.fit(x_full_scaled, y_full)
    pred_test = list(estimator_instance.predict(x_test_scaled))

    return pred_train, pred_val, pred_test


class LazyMIL:

    def __init__(self, task="regression", hopt=True, num_conf=10, num_cpu=20, output_folder=None, verbose=True):
        self.task = task
        self.hopt = hopt
        self.num_conf = num_conf
        self.output_folder = output_folder
        self.num_cpu = num_cpu
        self.verbose = verbose
        self.estimators_dict = REGRESSORS if self.task == "regression" else CLASSIFIERS

        if self.output_folder and os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)

    def run(self, df_train, df_val, df_test):

        # 1. Get data (smiles and prop)
        result_df_train = pd.DataFrame()
        smi_train, y_train = list(df_train.iloc[:, 0]), list(df_train.iloc[:, 1])
        result_df_train["SMILES"], result_df_train["Y_TRUE"] = smi_train, y_train

        result_df_val = pd.DataFrame()
        smi_val, y_val = list(df_val.iloc[:, 0]), list(df_val.iloc[:, 1])
        result_df_val["SMILES"], result_df_val["Y_TRUE"] = smi_val, y_val

        result_df_test = pd.DataFrame()
        smi_test, y_test = list(df_test.iloc[:, 0]), list(df_test.iloc[:, 1])
        result_df_test["SMILES"], result_df_test["Y_TRUE"] = smi_test, y_test

        # 2. Generate conformers
        conf_train = gen_conformers(smi_train, num_conf=self.num_conf, num_cpu=self.num_cpu, verbose=self.verbose)
        conf_val = gen_conformers(smi_val, num_conf=self.num_conf, num_cpu=self.num_cpu, verbose=self.verbose)
        conf_test = gen_conformers(smi_test, num_conf=self.num_conf, num_cpu=self.num_cpu, verbose=self.verbose)

        total_models = len(DESCRIPTORS) * len(self.estimators_dict)
        current_model = 0

        # 3. Calculate descriptors
        for desc_name, desc_calc in DESCRIPTORS.items():

            x_train = list(calc_descriptors(conf_train, desc_calc, verbose=False))
            x_val = list(calc_descriptors(conf_val, desc_calc, verbose=False))
            x_test = list(calc_descriptors(conf_test, desc_calc, verbose=False))

            # 4. Train models
            for est_name, estimator in self.estimators_dict.items():

                model_name = f"{desc_name}|{est_name}"
                current_model += 1

                start = time.time()
                with OutputSuppressor() as logger:
                    pred_train, pred_val, pred_test = run_in_subprocess(
                        build_model, x_train, x_val, x_test, y_train, y_val, y_test, estimator, self.hopt
                    )
                elapsed_min = (time.time() - start) / 60

                # 5. Write predictions
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

        return None
