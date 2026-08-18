import csv
import os
import types

import numpy as np
import pandas as pd
import pytest
from conftest import MockEstimator

import qsarmil.lazy as lazy_mod
from qsarmil.descriptor.rdkit import RDKitGEOM
from qsarmil.descriptor.wrapper import DescriptorWrapper
from qsarmil.lazy import (
    LazyMIL,
    build_model,
    calc_descriptors,
    clean_descriptors,
    compute_column_means,
    gen_conformers,
    scale_descriptors,
)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def test_gen_conformers_returns_ensembles():
    ensembles = gen_conformers(["CCO", "c1ccccc1"], num_conf=2, num_cpu=1, verbose=False)
    assert len(ensembles) == 2
    for ens in ensembles:
        assert len(ens) == 2


def test_gen_conformers_seed_affects_output():
    a = gen_conformers(["CC(C)Cc1ccc(cc1)C(C)C(=O)O"], num_conf=2, num_cpu=1, verbose=False, seed=42)
    b = gen_conformers(["CC(C)Cc1ccc(cc1)C(C)C(=O)O"], num_conf=2, num_cpu=1, verbose=False, seed=123)
    coords_a = a[0][0].GetConformer(0).GetPositions()
    coords_b = b[0][0].GetConformer(0).GetPositions()
    assert (coords_a != coords_b).any()


def test_compute_column_means():
    bags = [np.array([[10.0, 1.0]]), np.array([[20.0, 1.0]])]
    means = compute_column_means(bags)
    assert means[0] == 15.0
    assert means[1] == 1.0


def test_clean_descriptors_self_computed():
    bags = [np.array([[np.nan, 1.0], [10.0, 1.0]])]
    cleaned = clean_descriptors(bags)
    assert not np.isnan(cleaned[0]).any()
    assert cleaned[0][0, 0] == 10.0


def test_clean_descriptors_given_means():
    bags = [np.array([[np.nan, 1.0]])]
    given_means = np.array([99.0, 1.0])
    cleaned = clean_descriptors(bags, col_means=given_means)
    assert cleaned[0][0, 0] == 99.0


def test_calc_descriptors_real():
    conf_list = gen_conformers(["CCO", "c1ccccc1"], num_conf=2, num_cpu=1, verbose=False)
    calc = DescriptorWrapper(RDKitGEOM(), verbose=False)
    bags = calc_descriptors(conf_list, calc, verbose=False)
    assert len(bags) == 2
    assert bags[0].shape == (2, 11)

    train_means = compute_column_means(bags)
    bags_2 = calc_descriptors(conf_list, calc, verbose=False, col_means=train_means)
    assert len(bags_2) == 2


def test_scale_descriptors():
    x_train = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    x_test = [np.array([[2.0, 3.0]])]
    scaled_train, scaled_test = scale_descriptors(x_train, x_test)
    assert len(scaled_train) == 1
    assert len(scaled_test) == 1


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def _tiny_bags():
    x_train = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[2.0, 2.0]])]
    x_val = [np.array([[1.5, 2.5]])]
    x_test = [np.array([[1.2, 2.2]])]
    y_train = [1.0, 2.0]
    y_val = [1.5]
    y_test = [1.1]
    return x_train, x_val, x_test, y_train, y_val, y_test


def test_build_model_with_hopt():
    x_train, x_val, x_test, y_train, y_val, y_test = _tiny_bags()
    estimator = MockEstimator(supports_hopt=True)
    pred_train, pred_val, pred_test = build_model(
        x_train, x_val, x_test, y_train, y_val, y_test, estimator, hopt=True, seed=7
    )
    assert estimator.hopt_called is True
    assert len(pred_train) == 2
    assert len(pred_val) == 1
    assert len(pred_test) == 1


def test_build_model_without_hopt_attr():
    x_train, x_val, x_test, y_train, y_val, y_test = _tiny_bags()
    estimator = MockEstimator(supports_hopt=False)
    assert not hasattr(estimator, "hopt")
    pred_train, _, _ = build_model(
        x_train, x_val, x_test, y_train, y_val, y_test, estimator, hopt=True
    )
    assert len(pred_train) == 2


def test_build_model_hopt_false_skips_search():
    x_train, x_val, x_test, y_train, y_val, y_test = _tiny_bags()
    estimator = MockEstimator(supports_hopt=True)
    build_model(x_train, x_val, x_test, y_train, y_val, y_test, estimator, hopt=False)
    assert estimator.hopt_called is False


def test_lazy_estimator_factories_are_callable(monkeypatch):
    class DummyEstimator:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(
        lazy_mod,
        "import_module",
        lambda module_name: types.SimpleNamespace(Ridge=DummyEstimator, RidgeClassifier=DummyEstimator),
    )

    regressor_factories = lazy_mod._REGRESSORS()
    classifier_factories = lazy_mod._CLASSIFIERS()

    regressor = regressor_factories["Ridge"]()
    classifier = classifier_factories["RidgeClassifier"]()

    assert isinstance(regressor, DummyEstimator)
    assert isinstance(classifier, DummyEstimator)

def test_default_model_imports():
    from qsarmil.lazy import REGRESSORS
    for factory in REGRESSORS.values():
        est = factory()
        assert hasattr(est, "fit")
        assert hasattr(est, "predict")
    from qsarmil.lazy import CLASSIFIERS
    for factory in CLASSIFIERS.values():
        est = factory()
        assert hasattr(est, "fit")
        assert hasattr(est, "predict")

def test_default_descriptor_imports():
    from qsarmil.lazy import DESCRIPTORS
    for factory in DESCRIPTORS.values():
        desc = factory()
        assert hasattr(desc, "run")


def test_resolve_estimators_handles_callable_and_mapping():
    mapping = {"Mock": object()}
    assert lazy_mod.resolve_estimators(mapping) is mapping

    resolved = lazy_mod.resolve_estimators(lambda: mapping)
    assert resolved is mapping


def test_all_lazy_descriptors_resolve(monkeypatch):
    class DummyDescriptor:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class DummyModule:
        def __getattr__(self, name):
            return DummyDescriptor

    monkeypatch.setattr(lazy_mod, "import_module", lambda module_name: DummyModule())

    descriptor_keys = {
        "RDKitGEOM",
        "RDKitAUTOCORR",
        "RDKitRDF",
        "RDKitMORSE",
        "RDKitWHIM",
        "MolFeatUSRD",
        "MolFeatElectroShape",
        "RDKitGETAWAY",
        "MolFeatPmapper",
    }

    for name, factory in lazy_mod._DESCRIPTORS().items():
        assert name in descriptor_keys
        assert isinstance(factory(), DescriptorWrapper)


# ---------------------------------------------------------------------------
# LazyMIL.__init__
# ---------------------------------------------------------------------------

def test_lazymil_init_default_creates_temp_dir():
    lazy = LazyMIL()
    assert os.path.isdir(lazy.output_folder)


def test_lazymil_init_explicit_new_path(tmp_path):
    target = str(tmp_path / "new_output")
    lazy = LazyMIL(output_folder=target)
    assert lazy.output_folder == target
    assert os.path.isdir(target)


def test_lazymil_init_wipes_existing_path(tmp_path):
    target = str(tmp_path / "existing_output")
    os.makedirs(target)
    with open(os.path.join(target, "stale.txt"), "w") as f:
        f.write("stale")

    LazyMIL(output_folder=target)
    assert os.path.isdir(target)
    assert not os.path.exists(os.path.join(target, "stale.txt"))


# ---------------------------------------------------------------------------
# LazyMIL.run - full flow with fast monkeypatched descriptors/estimators
# ---------------------------------------------------------------------------

def _fast_descriptors():
    return {"RDKitGEOM": lambda: DescriptorWrapper(RDKitGEOM(), verbose=False)}


def test_lazymil_run_continuous_verbose(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    monkeypatch.setattr(lazy_mod, "REGRESSORS", {"Mock": MockEstimator(supports_hopt=False)})

    df_train = pd.DataFrame({0: ["CCO", "c1ccccc1", "not_a_valid_smiles!!!"], 1: [1.1, 2.2, 3.3]})
    df_val = pd.DataFrame({0: ["CCN", "CCC"], 1: [1.6, 2.6]})
    df_test = pd.DataFrame({0: ["CCCl", "CCF"], 1: [0.6, 1.6]})

    lazy = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=True)
    lazy.run(df_train, df_val, df_test)

    with open(tmp_path / "out" / "train.csv", newline="") as f:
        assert sum(1 for _ in csv.reader(f)) - 1 == 2  # the invalid SMILES got dropped

    captured = capsys.readouterr()
    assert "Running model:" in captured.out
    assert "Memory usage" in captured.out


def test_lazymil_run_binary_quiet(monkeypatch, tmp_path):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    monkeypatch.setattr(lazy_mod, "CLASSIFIERS", {"Mock": MockEstimator(supports_hopt=False)})

    df_train = pd.DataFrame({0: ["CCO", "c1ccccc1"], 1: [0, 1]})
    df_val = pd.DataFrame({0: ["CCN", "CCC"], 1: [0, 1]})
    df_test = pd.DataFrame({0: ["CCCl", "CCF"], 1: [0, 1]})

    lazy = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False)
    lazy.run(df_train, df_val, df_test)

    with open(tmp_path / "out" / "test.csv", newline="") as f:
        assert sum(1 for _ in csv.reader(f)) - 1 == 2


def test_lazymil_run_unsupported_task_type(tmp_path):
    df = pd.DataFrame({0: ["CCO", "c1ccccc1", "CCN", "CCC"], 1: ["a", "b", "c", "d"]})
    lazy = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False)
    with pytest.raises(ValueError, match="not supported"):
        lazy.run(df, df, df)
