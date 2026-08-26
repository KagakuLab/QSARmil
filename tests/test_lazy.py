import csv
import os

import numpy as np
import pandas as pd
import pytest

from conftest import MockEstimator
from milearn.wrapper import BagWrapper

import qsarmil.modelling.lazy as lazy_mod
from qsarmil.descriptor.rdkit import RDKitGEOM
from qsarmil.descriptor.wrapper import DescriptorWrapper
from qsarmil.utils.logging import FailedConformer, FailedMolecule
from qsarmil.modelling.lazy import (
    LazyMIL,
    build_model,
    calc_descriptors,
    gen_conformers,
    report_conformer_generation,
    report_smiles_parsing,
    scale_descriptors,
    target_fallback,
)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def test_gen_conformers_wraps_unparseable_smiles():
    results = gen_conformers(["CCO", "not_a_valid_smiles!!!"], num_conf=2, num_cpu=1, verbose=False)
    assert len(results) == 2
    assert isinstance(results[1], FailedMolecule)


def test_report_smiles_parsing_removes_invalid_rows(capsys):
    smiles = ["CCO", "not_a_valid_smiles!!!", "c1ccccc1"]
    confs = gen_conformers(smiles, num_conf=2, num_cpu=1, verbose=False)

    failed_idx = report_smiles_parsing(smiles, confs, verbose=True)

    assert failed_idx == {1}
    captured = capsys.readouterr()
    assert "Step-1. SMILES parsing" in captured.out
    assert "For 2 of 3 molecules, SMILES were parsed correctly." in captured.out
    assert "For 1 molecules, SMILES could not be parsed" in captured.out
    assert "not_a_valid_smiles!!!" in captured.out


def test_report_smiles_parsing_all_ok(capsys):
    smiles = ["CCO", "c1ccccc1"]
    confs = gen_conformers(smiles, num_conf=2, num_cpu=1, verbose=False)

    failed_idx = report_smiles_parsing(smiles, confs, verbose=True)

    assert failed_idx == set()
    captured = capsys.readouterr()
    assert "For 2 of 2 molecules, SMILES were parsed correctly." in captured.out
    assert "could not be parsed" not in captured.out


def test_report_smiles_parsing_quiet():
    smiles = ["not_a_valid_smiles!!!"]
    confs = gen_conformers(smiles, num_conf=2, num_cpu=1, verbose=False)
    failed_idx = report_smiles_parsing(smiles, confs, verbose=False)
    assert failed_idx == {0}


def test_report_conformer_generation_reports_stats_and_failures(capsys):
    smiles = ["CCO", "c1ccccc1", "not_a_valid_smiles!!!"]
    confs = gen_conformers(smiles, num_conf=2, num_cpu=1, verbose=False)
    parse_failed = report_smiles_parsing(smiles, confs, verbose=False)

    failed_idx = report_conformer_generation(smiles, confs, parse_failed, verbose=True)

    assert failed_idx == set()
    captured = capsys.readouterr()
    assert "Step-2. Conformer generation" in captured.out
    assert "For 2 of 2 molecules, conformers were generated successfully." in captured.out
    assert "Average num conf: 2.0 | min num conf: 2 | max num conf: 2" in captured.out


def test_report_conformer_generation_quiet():
    smiles = ["CCO"]
    confs = gen_conformers(smiles, num_conf=2, num_cpu=1, verbose=False)
    failed_idx = report_conformer_generation(smiles, confs, set(), verbose=False)
    assert failed_idx == set()


def test_report_conformer_generation_reports_embedding_failures(capsys):
    from qsarmil.utils.logging import FailedConformer

    smiles = ["CCO", "some_smiles_that_fails_embedding"]
    confs = gen_conformers(["CCO"], num_conf=2, num_cpu=1, verbose=False)
    confs.append(FailedConformer(None))  # simulate an embedding failure

    failed_idx = report_conformer_generation(smiles, confs, set(), verbose=True)

    assert failed_idx == {1}
    captured = capsys.readouterr()
    assert "For 1 of 2 molecules, conformers were generated successfully." in captured.out
    assert "For 1 molecules, conformer generation failed" in captured.out
    assert "Row 1:  some_smiles_that_fails_embedding" in captured.out


def test_target_fallback_continuous_is_mean():
    assert target_fallback([1.0, 2.0, 3.0], "continuous") == pytest.approx(2.0)


def test_target_fallback_binary_is_most_common_class():
    assert target_fallback([0, 1, 1, 1, 0], "binary") == 1


def test_subset_preserves_index_order():
    from qsarmil.modelling.lazy import _subset

    assert _subset(["a", "b", "c", "d"], [2, 0]) == ["c", "a"]


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


def test_calc_descriptors_real():
    conf_list = gen_conformers(["CCO", "c1ccccc1"], num_conf=2, num_cpu=1, verbose=False)
    calc = DescriptorWrapper(RDKitGEOM(), verbose=False)
    bags = calc_descriptors(conf_list, calc)
    assert len(bags) == 2
    assert bags[0].shape == (2, 11)

    bags_2 = calc_descriptors(conf_list, calc)
    assert len(bags_2) == 2


def test_calc_descriptors_suppresses_low_level_ticker(capsys):
    calc = DescriptorWrapper(lambda mol, **kw: np.array([1.0]), verbose=True)
    conf_list = [[object()], [object()]]

    calc_descriptors(conf_list, calc)

    assert calc.verbose is False
    assert capsys.readouterr().out == ""


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
    y_train = [1.0, 2.0]
    y_val = [1.5]
    return x_train, x_val, y_train, y_val


def test_build_model_with_hopt():
    x_train, x_val, y_train, y_val = _tiny_bags()
    estimator = MockEstimator(supports_hopt=True)
    pred_train, pred_val, fitted_estimator, fitted_scaler = build_model(
        x_train, x_val, y_train, y_val, estimator, hopt=True, seed=7
    )
    assert estimator.hopt_called is True
    assert len(pred_train) == 2
    assert len(pred_val) == 1
    assert hasattr(fitted_estimator, "predict")
    assert hasattr(fitted_scaler, "transform")


def test_build_model_forces_accelerator_into_hopt_grid():
    """The explicit accelerator must win over DEFAULT_PARAM_GRID's own fixed value."""
    x_train, x_val, y_train, y_val = _tiny_bags()
    estimator = MockEstimator(supports_hopt=True)
    build_model(x_train, x_val, y_train, y_val, estimator, hopt=True, accelerator="gpu")
    assert estimator.last_param_grid["accelerator"] == "gpu"


def test_build_model_without_hopt_attr():
    x_train, x_val, y_train, y_val = _tiny_bags()
    estimator = MockEstimator(supports_hopt=False)
    assert not hasattr(estimator, "hopt")
    pred_train, _, _, _ = build_model(x_train, x_val, y_train, y_val, estimator, hopt=True)
    assert len(pred_train) == 2


def test_build_model_hopt_false_skips_search():
    x_train, x_val, y_train, y_val = _tiny_bags()
    estimator = MockEstimator(supports_hopt=True)
    build_model(x_train, x_val, y_train, y_val, estimator, hopt=False)
    assert estimator.hopt_called is False


def test_build_model_with_sklearn_ridge_accepts_pooled_2d():
    from sklearn.linear_model import Ridge

    x_train, x_val, y_train, y_val = _tiny_bags()
    estimator = BagWrapper(Ridge())
    pred_train, pred_val, fitted_estimator, fitted_scaler = build_model(
        x_train, x_val, y_train, y_val, estimator, hopt=False
    )

    assert len(pred_train) == len(y_train)
    assert len(pred_val) == len(y_val)
    assert hasattr(fitted_estimator, "predict")
    assert hasattr(fitted_scaler, "transform")


def test_build_model_sklearn_ridge_accepts_pooled_2d():
    from sklearn.linear_model import Ridge

    x_train, x_val, y_train, y_val = _tiny_bags()
    estimator = BagWrapper(Ridge())
    pred_train, pred_val, fitted_estimator, fitted_scaler = build_model(
        x_train, x_val, y_train, y_val, estimator, hopt=False
    )

    assert len(pred_train) == len(y_train)
    assert len(pred_val) == len(y_val)
    assert hasattr(fitted_estimator, "predict")
    assert hasattr(fitted_scaler, "transform")


def test_build_model_hopt_path():
    x_train, x_val, y_train, y_val = _tiny_bags()
    estimator = MockEstimator(supports_hopt=True)
    build_model(x_train, x_val, y_train, y_val, estimator, hopt=True, seed=11)
    assert estimator.hopt_called is True


def test_default_model_imports():
    from qsarmil.modelling.lazy import REGRESSORS
    for factory in REGRESSORS.values():
        est = factory()
        assert hasattr(est, "fit")
        assert hasattr(est, "predict")
    from qsarmil.modelling.lazy import CLASSIFIERS
    for factory in CLASSIFIERS.values():
        est = factory()
        assert hasattr(est, "fit")
        assert hasattr(est, "predict")

def test_default_descriptor_imports():
    from qsarmil.modelling.lazy import DESCRIPTORS
    for factory in DESCRIPTORS.values():
        desc = factory()
        assert hasattr(desc, "run")

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


def test_model_factory_wraps_non_milearn_estimators(monkeypatch):
    class DummyEstimator:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit(self, x, y):
            return self

        def predict(self, x):
            return x

    class DummyModule:
        pass

    DummyModule.DummyEstimator = DummyEstimator

    monkeypatch.setattr(lazy_mod, "import_module", lambda module_name: DummyModule())

    factory = lazy_mod.model_factory("sklearn.dummy", "DummyEstimator", foo="bar")
    wrapped = factory()

    assert isinstance(wrapped, BagWrapper)


def test_model_factory_ignores_accelerator_for_non_milearn_estimators(monkeypatch):
    class DummyEstimator:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def fit(self, x, y):
            return self

        def predict(self, x):
            return x

    class DummyModule:
        pass

    DummyModule.DummyEstimator = DummyEstimator
    monkeypatch.setattr(lazy_mod, "import_module", lambda module_name: DummyModule())

    factory = lazy_mod.model_factory("sklearn.dummy", "DummyEstimator")
    wrapped = factory(accelerator="gpu")

    assert isinstance(wrapped, BagWrapper)
    # accelerator must never reach a non-milearn (e.g. sklearn) estimator's constructor
    assert "accelerator" not in wrapped.estimator.kwargs


def test_model_factory_passes_accelerator_to_milearn_estimators(monkeypatch):
    class DummyMilearnEstimator:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class DummyModule:
        pass

    DummyModule.DummyMilearnEstimator = DummyMilearnEstimator
    monkeypatch.setattr(lazy_mod, "import_module", lambda module_name: DummyModule())

    factory = lazy_mod.model_factory("milearn.network.regressor", "DummyMilearnEstimator")

    default_instance = factory()
    assert "accelerator" not in default_instance.kwargs

    gpu_instance = factory(accelerator="gpu")
    assert gpu_instance.kwargs["accelerator"] == "gpu"


# ---------------------------------------------------------------------------
# LazyMIL.__init__
# ---------------------------------------------------------------------------

def test_lazymil_init_default_creates_temp_dir():
    lazy = LazyMIL()
    assert os.path.isdir(lazy.output_folder)


def test_lazymil_accelerator_defaults_to_cpu():
    lazy = LazyMIL()
    assert lazy.accelerator == "cpu"


def test_lazymil_accelerator_explicit_override(tmp_path):
    lazy = LazyMIL(output_folder=str(tmp_path / "out"), accelerator="gpu")
    assert lazy.accelerator == "gpu"


def test_lazymil_accelerator_rejects_invalid_value(tmp_path):
    with pytest.raises(ValueError, match="cpu.*gpu"):
        LazyMIL(output_folder=str(tmp_path / "out"), accelerator="auto")


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

class _IdentityScaler:
    """Module-level (picklable) stand-in for BagMinMaxScaler."""

    def transform(self, x):
        return x


class _BadEstimator:
    """Module-level (picklable) estimator that always raises an unrelated AttributeError."""

    def predict(self, x):
        raise AttributeError("different attribute error")


def _fast_descriptors():
    return {"RDKitGEOM": lambda: DescriptorWrapper(RDKitGEOM(), verbose=False)}


def test_lazymil_run_continuous_verbose(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    monkeypatch.setattr(lazy_mod, "REGRESSORS", {"Mock": MockEstimator(supports_hopt=False)})

    smiles = ["CCO", "c1ccccc1", "not_a_valid_smiles!!!", "CCN", "CCC", "CCCl"]
    y = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    lazy = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=True)
    lazy.run(smiles, y)

    with open(tmp_path / "out" / "train.csv", newline="") as f, open(
        tmp_path / "out" / "val.csv", newline=""
    ) as fv:
        n_train = sum(1 for _ in csv.reader(f)) - 1
        n_val = sum(1 for _ in csv.reader(fv)) - 1
    assert n_train + n_val == 5  # the invalid SMILES got dropped, the rest got split

    captured = capsys.readouterr()
    assert "Step-1. SMILES parsing" in captured.out
    assert "Step-2. Conformer generation" in captured.out
    assert "Step-3. Descriptor calculation" in captured.out
    assert "Step-4. Individual model building" in captured.out
    assert "not_a_valid_smiles!!!" in captured.out
    assert "[1/1] RDKitGEOM:" in captured.out
    assert "[1/1] RDKitGEOM|Mock" in captured.out
    assert "Finished in" in captured.out
    assert "Memory usage:" in captured.out


def test_lazymil_run_binary_quiet(monkeypatch, tmp_path):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    monkeypatch.setattr(lazy_mod, "CLASSIFIERS", {"Mock": MockEstimator(supports_hopt=False)})

    smiles = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y = [0, 1, 0, 1, 0]
    lazy = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False)
    lazy.run(smiles, y)


def test_lazymil_run_unsupported_task_type(tmp_path):
    smiles = ["CCO", "c1ccccc1", "CCN", "CCC"]
    y = ["a", "b", "c", "d"]
    lazy = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False)
    with pytest.raises(ValueError, match="not supported"):
        lazy.run(smiles, y)


def test_lazymil_run_forced_task_skips_autodetection(monkeypatch, tmp_path):
    """Passing task= bypasses type_of_target, so a 2-value numeric target
    (which sklearn would call 'binary') still routes to REGRESSORS."""
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    monkeypatch.setattr(lazy_mod, "REGRESSORS", {"Mock": MockEstimator(supports_hopt=False)})
    monkeypatch.setattr(lazy_mod, "CLASSIFIERS", {})

    smiles = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y = [1.0, 3.0, 1.0, 3.0, 1.0]  # only 2 distinct values

    lazy = LazyMIL(
        hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False, task="continuous"
    )
    lazy.run(smiles, y)

    assert lazy._task_type == "continuous"
    assert "RDKitGEOM|Mock" in lazy._trained_models


def test_lazymil_run_splits_reproducibly_with_seed(monkeypatch, tmp_path):
    """self.seed reaches the internal train/val split."""
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    monkeypatch.setattr(lazy_mod, "REGRESSORS", {"Mock": MockEstimator(supports_hopt=False)})

    smiles = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y = [1.1, 2.2, 3.3, 4.4, 5.5]

    lazy_a = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out_a"), verbose=False, seed=1)
    lazy_b = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out_b"), verbose=False, seed=2)
    lazy_a.run(smiles, y)
    lazy_b.run(smiles, y)

    train_a = pd.read_csv(tmp_path / "out_a" / "train.csv")
    train_b = pd.read_csv(tmp_path / "out_b" / "train.csv")
    assert list(train_a["SMILES"]) != list(train_b["SMILES"])


def test_lazymil_run_threads_accelerator_into_estimator_construction(monkeypatch, tmp_path):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    mock = MockEstimator(supports_hopt=False)
    monkeypatch.setattr(lazy_mod, "REGRESSORS", {"Mock": mock})

    smiles = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y = [1.1, 2.2, 3.3, 4.4, 5.5]
    lazy = LazyMIL(
        hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False, accelerator="gpu"
    )
    lazy.run(smiles, y)

    assert mock.accelerator == "gpu"


def test_lazymil_run_stores_mean_fallback_for_regression(monkeypatch, tmp_path):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    monkeypatch.setattr(lazy_mod, "REGRESSORS", {"Mock": MockEstimator(supports_hopt=False)})

    smiles = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y = [1.1, 3.3, 2.2, 4.4, 5.5]
    lazy = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False)
    lazy.run(smiles, y)

    assert lazy._train_fallback == pytest.approx(sum(y) / len(y))


def test_lazymil_run_stores_mode_fallback_for_classification(monkeypatch, tmp_path):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    monkeypatch.setattr(lazy_mod, "CLASSIFIERS", {"Mock": MockEstimator(supports_hopt=False)})

    smiles = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y = [0, 1, 1, 1, 0]
    lazy = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False)
    lazy.run(smiles, y)

    assert lazy._train_fallback == 1


def test_lazymil_predict_before_train_raises(tmp_path):
    lazy = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False)
    with pytest.raises(RuntimeError, match="not trained"):
        lazy.predict(["CCO"])


def test_lazymil_save_load_predict_inference_only(monkeypatch, tmp_path):
    """A trained LazyMIL is a plain picklable object - no dedicated save()/load() needed."""
def test_lazymil_predict_reuses_same_process_trained_models(monkeypatch, tmp_path):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    monkeypatch.setattr(lazy_mod, "REGRESSORS", {"Mock": MockEstimator(supports_hopt=False)})

    smiles = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y = [1.1, 2.2, 3.3, 4.4, 5.5]
    lazy = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False)
    lazy.run(smiles, y)
    assert lazy.is_trained is True

    def _should_not_retrain(*args, **kwargs):
        raise AssertionError("predict path retrained a model")

    monkeypatch.setattr(lazy_mod, "build_model", _should_not_retrain)

    pred_df = lazy.predict(["CCCl", "CCF"], save=True)
    assert len(pred_df) == 2
    assert "RDKitGEOM|Mock" in pred_df.columns
    assert os.path.exists(tmp_path / "out" / "test.csv")


def test_lazymil_predict_imputes_failed_molecules_with_fallback(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    monkeypatch.setattr(lazy_mod, "REGRESSORS", {"Mock": MockEstimator(supports_hopt=False)})

    smiles = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y = [1.1, 3.3, 2.2, 4.4, 5.5]
    lazy = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=True)
    lazy.run(smiles, y)

    pred_df = lazy.predict(["CCO", "not_a_valid_smiles!!!"])

    assert len(pred_df) == 2
    assert pred_df["RDKitGEOM|Mock"].iloc[1] == pytest.approx(lazy._train_fallback)

    captured = capsys.readouterr()
    assert "1 molecule(s) could not be processed" in captured.out
    assert "not_a_valid_smiles!!!" in captured.out


def test_lazymil_predict_silent_when_nothing_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    monkeypatch.setattr(lazy_mod, "REGRESSORS", {"Mock": MockEstimator(supports_hopt=False)})

    smiles = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y = [1.1, 2.2, 3.3, 4.4, 5.5]
    lazy = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=True)
    lazy.run(smiles, y)
    capsys.readouterr()  # discard training output

    lazy.predict(["CCO"])

    # predict() should stay silent unless a molecule actually fails.
    assert capsys.readouterr().out == ""


def test_lazymil_predict_reraises_other_attributeerror(monkeypatch, tmp_path):
    monkeypatch.setattr(lazy_mod, "gen_conformers", lambda *args, **kwargs: [[object()]])
    monkeypatch.setattr(lazy_mod, "calc_descriptors", lambda *args, **kwargs: [np.array([[0.1, 0.2]])])

    lazy = LazyMIL(hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False)
    lazy._fitted_descriptors = {"RDKitGEOM": DescriptorWrapper(RDKitGEOM(), verbose=False)}
    lazy._trained_models = {
        "RDKitGEOM|Mock": {"descriptor": "RDKitGEOM", "estimator": _BadEstimator(), "scaler": _IdentityScaler()}
    }

    with pytest.raises(AttributeError, match="different attribute error"):
        lazy.predict(["CCO"])


def test_print_progress_item(capsys):
    from qsarmil.modelling.lazy import _print_progress_item

    _print_progress_item(1, 72, "RDKitGEOM|Mock", elapsed_min=7.876, mem_gb=1.2549)
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "[1/72] RDKitGEOM|Mock"
    assert lines[1] == "       > Finished in 7.88 min | Memory usage: 1.255 G"
    # The ">" on the second line lines up under where the label starts.
    assert lines[1].index(">") == lines[0].index("R")
