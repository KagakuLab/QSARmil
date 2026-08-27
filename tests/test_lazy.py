import os

import numpy as np
import pytest
from conftest import MockEstimator

import qsarmil.modelling.lazy as lazy_mod
from qsarmil.descriptor.rdkit import RDKitGEOM
from qsarmil.descriptor.wrapper import DescriptorWrapper
from qsarmil.modelling.lazy import (
    LazyMIL,
    calculate_descriptors,
    generate_conformers,
    parse_smiles,
    scale_descriptors,
    baseline_prediction,
    train_estimator,
)
from qsarmil.utils.logging import FailedMolecule

# ---------------------------------------------------------------------------
# parse_smiles
# ---------------------------------------------------------------------------

def test_parse_smiles_wraps_unparseable_smiles():
    results = parse_smiles(["CCO", "not_a_valid_smiles!!!"])
    assert len(results) == 2
    assert results[0] is not None and not isinstance(results[0], FailedMolecule)
    assert isinstance(results[1], FailedMolecule)


def test_parse_smiles_verbose_reports_summary(capsys):
    parse_smiles(["CCO", "not_a_valid_smiles!!!", "c1ccccc1"], verbose=True)
    captured = capsys.readouterr()
    assert "Parsed 2 of 3 SMILES successfully." in captured.out


def test_parse_smiles_quiet_prints_nothing(capsys):
    parse_smiles(["CCO", "not_a_valid_smiles!!!"], verbose=False)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# generate_conformers
# ---------------------------------------------------------------------------

def test_generate_conformers_returns_ensembles():
    mols = parse_smiles(["CCO", "c1ccccc1"])
    ensembles = generate_conformers(mols, num_conf=2, num_cpu=1)
    assert len(ensembles) == 2
    for ens in ensembles:
        assert len(ens) == 2


def test_generate_conformers_seed_affects_output():
    mols = parse_smiles(["CC(C)Cc1ccc(cc1)C(C)C(=O)O"])
    a = generate_conformers(mols, num_conf=2, num_cpu=1, seed=42)
    b = generate_conformers(mols, num_conf=2, num_cpu=1, seed=123)
    coords_a = a[0][0].GetConformer(0).GetPositions()
    coords_b = b[0][0].GetConformer(0).GetPositions()
    assert (coords_a != coords_b).any()


def test_generate_conformers_passes_through_failed_molecule():
    mols = parse_smiles(["CCO", "not_a_valid_smiles!!!"])
    confs = generate_conformers(mols, num_conf=2, num_cpu=1)
    assert isinstance(confs[0], list)
    assert isinstance(confs[1], FailedMolecule)


def test_generate_conformers_verbose_reports_summary(capsys):
    mols = parse_smiles(["CCO", "not_a_valid_smiles!!!"])
    generate_conformers(mols, num_conf=2, num_cpu=1, verbose=True)
    captured = capsys.readouterr()
    assert "Generated conformers for 1 of 2 molecules." in captured.out


def test_generate_conformers_quiet_prints_nothing(capsys):
    mols = parse_smiles(["CCO"])
    generate_conformers(mols, num_conf=2, num_cpu=1, verbose=False)
    assert capsys.readouterr().out == ""


def test_generate_conformers_suppresses_low_level_ticker(capsys):
    """The RDKit-level per-molecule ticker stays off regardless of `verbose`, since LazyMIL owns
    its own step-level progress instead."""
    mols = parse_smiles(["CCO"])
    generate_conformers(mols, num_conf=2, num_cpu=1, verbose=True)
    assert "Generating conformers:" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# calculate_descriptors / scale_descriptors
# ---------------------------------------------------------------------------

def test_calculate_descriptors_real():
    mols = parse_smiles(["CCO", "c1ccccc1"])
    conf_list = generate_conformers(mols, num_conf=2, num_cpu=1)
    calc = DescriptorWrapper(RDKitGEOM(), verbose=False)
    bags = calculate_descriptors(conf_list, calc)
    assert len(bags) == 2
    assert bags[0].shape == (2, 11)

    bags_2 = calculate_descriptors(conf_list, calc)
    assert len(bags_2) == 2


def test_calculate_descriptors_suppresses_low_level_ticker(capsys):
    calc = DescriptorWrapper(lambda mol, **kw: np.array([1.0]), verbose=True)
    conf_list = [[object()], [object()]]

    calculate_descriptors(conf_list, calc)

    assert calc.verbose is False
    assert capsys.readouterr().out == ""


def test_scale_descriptors():
    x_train = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    x_test = [np.array([[2.0, 3.0]])]
    scaled_train, scaled_test = scale_descriptors(x_train, x_test)
    assert len(scaled_train) == 1
    assert len(scaled_test) == 1


def test_scale_descriptors_handles_empty_test():
    x_train = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    scaled_train, scaled_test = scale_descriptors(x_train, [])
    assert len(scaled_train) == 1
    assert list(scaled_test) == []


# ---------------------------------------------------------------------------
# target_fallback
# ---------------------------------------------------------------------------

def test_target_fallback_continuous_is_mean():
    assert baseline_prediction([1.0, 2.0, 3.0], "continuous") == pytest.approx(2.0)


def test_target_fallback_binary_is_most_common_class():
    assert baseline_prediction([0, 1, 1, 1, 0], "binary") == 1


# ---------------------------------------------------------------------------
# train_estimator
# ---------------------------------------------------------------------------

def _tiny_bags():
    x_train = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[2.0, 2.0]])]
    x_val = [np.array([[1.5, 2.5]])]
    x_test = [np.array([[1.2, 2.2]])]
    y_train = [1.0, 2.0]
    y_val = [1.5]
    return x_train, x_val, x_test, y_train, y_val


def test_train_estimator_with_hopt():
    x_train, x_val, x_test, y_train, y_val = _tiny_bags()
    estimator = MockEstimator(supports_hopt=True)
    pred_train, pred_val, pred_test = train_estimator(
        x_train, x_val, x_test, y_train, y_val, estimator, hopt=True, seed=7
    )
    assert estimator.hopt_called is True
    assert len(pred_train) == 2
    assert len(pred_val) == 1
    assert len(pred_test) == 1


def test_train_estimator_without_hopt_attr():
    x_train, x_val, x_test, y_train, y_val = _tiny_bags()
    estimator = MockEstimator(supports_hopt=False)
    assert not hasattr(estimator, "hopt")
    pred_train, _, _ = train_estimator(x_train, x_val, x_test, y_train, y_val, estimator, hopt=True)
    assert len(pred_train) == 2


def test_train_estimator_hopt_false_skips_search():
    x_train, x_val, x_test, y_train, y_val = _tiny_bags()
    estimator = MockEstimator(supports_hopt=True)
    train_estimator(x_train, x_val, x_test, y_train, y_val, estimator, hopt=False)
    assert estimator.hopt_called is False


def test_train_estimator_empty_test_returns_no_predictions():
    x_train, x_val, _, y_train, y_val = _tiny_bags()
    estimator = MockEstimator(supports_hopt=False)
    _, _, pred_test = train_estimator(x_train, x_val, [], y_train, y_val, estimator, hopt=False)
    assert pred_test == []


def test_train_estimator_with_sklearn_ridge_accepts_pooled_2d():
    from milearn.wrapper import BagWrapper
    from sklearn.linear_model import Ridge

    x_train, x_val, x_test, y_train, y_val = _tiny_bags()
    estimator = BagWrapper(Ridge())
    pred_train, pred_val, pred_test = train_estimator(
        x_train, x_val, x_test, y_train, y_val, estimator, hopt=False
    )

    assert len(pred_train) == len(y_train)
    assert len(pred_val) == len(y_val)
    assert len(pred_test) == len(x_test)


def test_train_estimator_hopt_path():
    x_train, x_val, x_test, y_train, y_val = _tiny_bags()
    estimator = MockEstimator(supports_hopt=True)
    train_estimator(x_train, x_val, x_test, y_train, y_val, estimator, hopt=True, seed=11)
    assert estimator.hopt_called is True


def test_train_estimator_forces_accelerator_into_hopt_grid():
    """The explicit accelerator must win over HYPERPARAMETERS' own fixed value."""
    x_train, x_val, x_test, y_train, y_val = _tiny_bags()
    estimator = MockEstimator(supports_hopt=True)
    train_estimator(x_train, x_val, x_test, y_train, y_val, estimator, hopt=True, accelerator="gpu")
    assert estimator.last_param_grid["accelerator"] == "gpu"


# ---------------------------------------------------------------------------
# DESCRIPTORS / REGRESSORS / CLASSIFIERS - built-in factory dicts
# ---------------------------------------------------------------------------

def test_default_descriptor_factories_resolve():
    from qsarmil.modelling.lazy import DESCRIPTORS

    for factory in DESCRIPTORS.values():
        desc = factory()
        assert hasattr(desc, "run")


def test_default_descriptor_factories_are_independent_instances():
    """Each call must build a fresh DescriptorWrapper, so separate LazyMIL.run() calls
    (or separate descriptors within one run) never share fitted state."""
    from qsarmil.modelling.lazy import DESCRIPTORS

    factory = DESCRIPTORS["RDKitGEOM"]
    assert factory() is not factory()


def test_default_regressor_and_classifier_factories_resolve():
    from qsarmil.modelling.lazy import CLASSIFIERS, REGRESSORS

    for factory in REGRESSORS.values():
        est = factory()
        assert hasattr(est, "fit")
        assert hasattr(est, "predict")
    for factory in CLASSIFIERS.values():
        est = factory()
        assert hasattr(est, "fit")
        assert hasattr(est, "predict")


def test_default_regressor_factories_are_independent_instances():
    """Same reasoning as descriptors: one shared estimator object being retrained/overwritten
    across every descriptor type would silently corrupt the results."""
    from qsarmil.modelling.lazy import REGRESSORS

    factory = REGRESSORS["MeanBagNetworkRegressor"]
    assert factory() is not factory()


# ---------------------------------------------------------------------------
# LazyMIL.__init__
# ---------------------------------------------------------------------------

def test_lazymil_init_default_creates_temp_dir():
    lazy = LazyMIL(task="continuous")
    assert os.path.isdir(lazy.output_folder)


def test_lazymil_accelerator_defaults_to_cpu():
    lazy = LazyMIL(task="continuous")
    assert lazy.accelerator == "cpu"


def test_lazymil_accelerator_explicit_override(tmp_path):
    lazy = LazyMIL(task="continuous", output_folder=str(tmp_path / "out"), accelerator="gpu")
    assert lazy.accelerator == "gpu"


def test_lazymil_init_sets_estimators_from_task():
    lazy_reg = LazyMIL(task="continuous")
    assert lazy_reg.ESTIMATORS is lazy_mod.REGRESSORS

    lazy_clf = LazyMIL(task="binary")
    assert lazy_clf.ESTIMATORS is lazy_mod.CLASSIFIERS


def test_lazymil_init_explicit_new_path(tmp_path):
    target = str(tmp_path / "new_output")
    lazy = LazyMIL(task="continuous", output_folder=target)
    assert lazy.output_folder == target
    assert os.path.isdir(target)


def test_lazymil_init_wipes_existing_path(tmp_path):
    target = str(tmp_path / "existing_output")
    os.makedirs(target)
    with open(os.path.join(target, "stale.txt"), "w") as f:
        f.write("stale")

    LazyMIL(task="continuous", output_folder=target)
    assert os.path.isdir(target)
    assert not os.path.exists(os.path.join(target, "stale.txt"))


# ---------------------------------------------------------------------------
# LazyMIL.run - full flow with fast monkeypatched descriptors/estimators
# ---------------------------------------------------------------------------

def _fast_descriptors():
    return {"RDKitGEOM": lambda: DescriptorWrapper(RDKitGEOM(), verbose=False)}


def test_lazymil_run_continuous_verbose(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())

    smi_train = ["CCO", "c1ccccc1", "not_a_valid_smiles!!!", "CCN"]
    y_train = [1.1, 2.2, 3.3, 4.4]
    smi_val = ["CCC"]
    y_val = [5.5]
    smi_test = ["CCCl"]

    lazy = LazyMIL(
        task="continuous", hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=True
    )
    lazy.ESTIMATORS = {"Mock": MockEstimator(supports_hopt=False)}
    result_train, result_val, result_test = lazy.run(smi_train, y_train, smi_val, y_val, smi_test)

    assert len(result_train) == 3  # the invalid SMILES got dropped
    assert len(result_val) == 1
    assert len(result_test) == 1  # test rows are never dropped
    assert "RDKitGEOM|Mock" in result_test.columns

    captured = capsys.readouterr()
    assert "Step-1. SMILES parsing" in captured.out
    assert "Step-2. Conformer generation" in captured.out
    assert "Step-3. Descriptor calculation" in captured.out
    assert "Step-4. Model training" in captured.out
    assert "Parsed 5 of 6 SMILES successfully." in captured.out
    assert "RDKitGEOM: done" in captured.out
    assert "[1/1] RDKitGEOM|Mock" in captured.out
    assert (tmp_path / "out" / "train.csv").exists()
    assert (tmp_path / "out" / "val.csv").exists()
    assert (tmp_path / "out" / "test.csv").exists()


def test_lazymil_run_binary_quiet(monkeypatch, tmp_path):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())

    smi_train = ["CCO", "c1ccccc1", "CCN", "CCC"]
    y_train = [0, 1, 0, 1]
    smi_val = ["CCCl"]
    y_val = [0]

    lazy = LazyMIL(
        task="binary", hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False
    )
    lazy.ESTIMATORS = {"Mock": MockEstimator(supports_hopt=False)}
    lazy.run(smi_train, y_train, smi_val, y_val, ["CCF"])


def test_lazymil_run_threads_accelerator_into_estimator_construction(monkeypatch, tmp_path):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    mock = MockEstimator(supports_hopt=False)

    smi_train = ["CCO", "c1ccccc1", "CCN", "CCC"]
    y_train = [1.1, 2.2, 3.3, 4.4]
    smi_val = ["CCCl"]
    y_val = [5.5]

    lazy = LazyMIL(
        task="continuous", hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"),
        verbose=False, accelerator="gpu",
    )
    lazy.ESTIMATORS = {"Mock": mock}
    lazy.run(smi_train, y_train, smi_val, y_val, ["CCF"])

    assert mock.accelerator == "gpu"


def test_lazymil_run_splits_are_taken_as_given(monkeypatch, tmp_path):
    """LazyMIL no longer performs its own train/val split - it trusts the split it's handed."""
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())

    smi_train = ["CCO", "c1ccccc1", "CCN", "CCC"]
    y_train = [1.1, 2.2, 3.3, 4.4]
    smi_val = ["CCCl", "CCF"]
    y_val = [5.5, 6.6]

    lazy = LazyMIL(task="continuous", hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False)
    lazy.ESTIMATORS = {"Mock": MockEstimator(supports_hopt=False)}
    result_train, result_val, _ = lazy.run(smi_train, y_train, smi_val, y_val, [])

    assert list(result_train["SMILES"]) == smi_train
    assert list(result_val["SMILES"]) == smi_val


def test_lazymil_run_descriptors_calculated_once_across_splits(monkeypatch, tmp_path):
    """Descriptors are computed for train+val+test together in one call per descriptor type,
    not once per split."""
    calls = []

    real_calculate_descriptors = lazy_mod.calculate_descriptors

    def _counting_calculate_descriptors(conf_list, calculator):
        calls.append(len(conf_list))
        return real_calculate_descriptors(conf_list, calculator)

    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())
    monkeypatch.setattr(lazy_mod, "calculate_descriptors", _counting_calculate_descriptors)

    smi_train = ["CCO", "c1ccccc1", "CCN"]
    y_train = [1.1, 2.2, 3.3]
    smi_val = ["CCC"]
    y_val = [4.4]
    smi_test = ["CCCl", "CCF"]

    lazy = LazyMIL(task="continuous", hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False)
    lazy.ESTIMATORS = {"Mock": MockEstimator(supports_hopt=False)}
    lazy.run(smi_train, y_train, smi_val, y_val, smi_test)

    # one call, covering every molecule across all three splits
    assert calls == [len(smi_train) + len(smi_val) + len(smi_test)]


def test_lazymil_run_test_predictions_use_fallback_for_failed_molecules(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())

    smi_train = ["CCO", "c1ccccc1", "CCN", "CCC"]
    y_train = [1.1, 3.3, 2.2, 4.4]
    smi_val = ["CCCl"]
    y_val = [5.5]
    smi_test = ["CCF", "not_a_valid_smiles!!!"]

    lazy = LazyMIL(
        task="continuous", hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=True
    )
    lazy.ESTIMATORS = {"Mock": MockEstimator(supports_hopt=False)}
    _, _, result_test = lazy.run(smi_train, y_train, smi_val, y_val, smi_test)

    assert len(result_test) == 2  # test rows are never dropped, even on failure
    fallback = baseline_prediction(y_train, "continuous")
    assert result_test["RDKitGEOM|Mock"].iloc[1] == pytest.approx(fallback)

    captured = capsys.readouterr()
    assert "1 test molecule(s) could not be processed" in captured.out


def test_lazymil_run_test_predictions_silent_when_nothing_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())

    smi_train = ["CCO", "c1ccccc1", "CCN", "CCC"]
    y_train = [1.1, 2.2, 3.3, 4.4]
    smi_val = ["CCCl"]
    y_val = [5.5]

    lazy = LazyMIL(
        task="continuous", hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=True
    )
    lazy.ESTIMATORS = {"Mock": MockEstimator(supports_hopt=False)}
    lazy.run(smi_train, y_train, smi_val, y_val, ["CCF"])

    assert "could not be processed" not in capsys.readouterr().out


def test_lazymil_run_empty_test_set(monkeypatch, tmp_path):
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", _fast_descriptors())

    lazy = LazyMIL(task="continuous", hopt=False, num_conf=2, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False)
    lazy.ESTIMATORS = {"Mock": MockEstimator(supports_hopt=False)}
    _, _, result_test = lazy.run(
        ["CCO", "c1ccccc1", "CCN", "CCC"], [1.1, 2.2, 3.3, 4.4], ["CCCl"], [5.5], []
    )
    assert len(result_test) == 0
    assert list(result_test.columns) == ["SMILES", "RDKitGEOM|Mock"]
