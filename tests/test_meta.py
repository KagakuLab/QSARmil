import os
import pickle
from typing import Any, cast

import pandas as pd
import pytest
from conftest import MockEstimator

import qsarmil.lazy as lazy_mod
import qsarmil.meta as meta_mod
from qsarmil.descriptor.rdkit import RDKitGEOM
from qsarmil.descriptor.wrapper import DescriptorWrapper
from qsarmil.meta import MultiConformerClassifier, MultiConformerEstimator, MultiConformerRegressor


class FakeGeneticSearch:
    """Stand-in for qsarcons.consensus.GeneticSearch - real GA search is
    genuinely slow (tens of minutes even on tiny data) and isn't
    injectable, so this substitutes qsarmil's external collaborator at the
    test boundary while exercising all of qsarmil's own orchestration code."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def run(self, x_val, true_val):
        return list(x_val.columns)

    def predict(self, x_subset):
        return list(x_subset.mean(axis=1))


class UnpicklableConsensus:
    def __getstate__(self):
        raise pickle.PickleError("cannot pickle")

    def predict(self, x_subset):
        return list(x_subset.mean(axis=1))


def _patch_fast_pipeline(monkeypatch, classifier=False):
    monkeypatch.setattr(meta_mod, "GeneticSearch", FakeGeneticSearch)
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", {"RDKitGEOM": lambda: DescriptorWrapper(RDKitGEOM(), verbose=False)})
    if classifier:
        monkeypatch.setattr(lazy_mod, "CLASSIFIERS", {"Mock": MockEstimator(supports_hopt=False)})
    else:
        monkeypatch.setattr(lazy_mod, "REGRESSORS", {"Mock": MockEstimator(supports_hopt=False)})


def test_regressor_and_classifier_share_base_class():
    assert issubclass(MultiConformerRegressor, MultiConformerEstimator)
    assert issubclass(MultiConformerClassifier, MultiConformerEstimator)


def test_regressor_rejects_integer_target(tmp_path):
    model = MultiConformerRegressor(output_folder=str(tmp_path / "out"), verbose=False)
    with pytest.raises(ValueError, match="classification labels"):
        model.train(["CCO", "c1ccccc1", "CCN", "CCC"], [0, 1, 0, 1])


def test_regressor_rejects_boolean_target(tmp_path):
    model = MultiConformerRegressor(output_folder=str(tmp_path / "out"), verbose=False)
    with pytest.raises(ValueError, match="classification labels"):
        model.train(["CCO", "c1ccccc1", "CCN", "CCC"], [True, False, True, False])


def test_regressor_rejects_non_numeric_target(tmp_path):
    model = MultiConformerRegressor(output_folder=str(tmp_path / "out"), verbose=False)
    with pytest.raises(ValueError, match="not numeric"):
        model.train(["CCO", "c1ccccc1", "CCN", "CCC"], ["active", "inactive", "active", "inactive"])


def test_regressor_accepts_float_target_with_few_distinct_values(monkeypatch, tmp_path):
    """The whole point of _check_continuous_target being dtype-based, not
    cardinality-based: a float target with only 2 distinct values must NOT
    be rejected, since that's a legitimate (if small) regression dataset."""
    _patch_fast_pipeline(monkeypatch)
    model = MultiConformerRegressor(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False
    )
    model.train(["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"], [1.0, 3.0, 1.0, 3.0, 1.0])
    assert model.is_trained is True


def test_classifier_does_not_apply_continuous_target_check(monkeypatch, tmp_path):
    """MultiConformerClassifier has no analogous guard - integer labels are
    exactly what it expects."""
    _patch_fast_pipeline(monkeypatch, classifier=True)
    model = MultiConformerClassifier(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False
    )
    model.train(["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"], [0, 1, 0, 1, 0])
    assert model.is_trained is True


def test_regressor_forces_continuous_despite_two_value_target(monkeypatch, tmp_path):
    """Regression's whole reason to exist: sklearn's type_of_target treats a
    2-distinct-value numeric target as 'binary', which would otherwise route
    this into CLASSIFIERS instead of REGRESSORS."""
    monkeypatch.setattr(meta_mod, "GeneticSearch", FakeGeneticSearch)
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", {"RDKitGEOM": lambda: DescriptorWrapper(RDKitGEOM(), verbose=False)})
    monkeypatch.setattr(lazy_mod, "REGRESSORS", {"Mock": MockEstimator(supports_hopt=False)})
    monkeypatch.setattr(lazy_mod, "CLASSIFIERS", {})  # would raise KeyError-ish if wrongly routed here

    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.0, 3.0, 1.0, 3.0, 1.0]  # only 2 distinct values

    model = MultiConformerRegressor(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False
    )
    model.train(smiles_train, y_train)

    assert model.is_trained is True
    assert model._lazy_model._task_type == "continuous"


def test_classifier_forces_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(meta_mod, "GeneticSearch", FakeGeneticSearch)
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", {"RDKitGEOM": lambda: DescriptorWrapper(RDKitGEOM(), verbose=False)})
    monkeypatch.setattr(lazy_mod, "CLASSIFIERS", {"Mock": MockEstimator(supports_hopt=False)})

    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [0, 1, 0, 1, 0]

    model = MultiConformerClassifier(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False
    )
    model.train(smiles_train, y_train)

    assert model.is_trained is True
    assert model._lazy_model._task_type == "binary"


def test_init_default_creates_temp_dir():
    model = MultiConformerRegressor()
    assert os.path.isdir(model.output_folder)
    assert model.seed == 42


def test_train_then_predict_split_api(monkeypatch, tmp_path, capsys):
    _patch_fast_pipeline(monkeypatch)

    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.1, 2.2, 3.3, 4.4, 5.5]
    smiles_test = ["CCF"]

    model = MultiConformerRegressor(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=True, seed=42
    )
    model.train(smiles_train, y_train)

    assert model.is_trained is True
    assert len(model.best_consensus) > 0

    captured = capsys.readouterr()
    assert "Step-5. Genetic model consensus search" in captured.out
    assert "Best genetic consensus" in captured.out

    preds = model.predict(smiles_test)
    assert isinstance(preds, list)
    assert len(preds) == 1


def test_train_then_predict_binary_quiet(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch, classifier=True)

    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [0, 1, 0, 1, 0]
    smiles_test = ["CCF"]

    model = MultiConformerClassifier(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False, seed=99
    )
    model.train(smiles_train, y_train)
    preds = model.predict(smiles_test)

    assert isinstance(preds, list)
    assert len(preds) == 1


def test_seeds_produce_different_train_val_splits(monkeypatch, tmp_path):
    """self.seed reaches train_test_split's random_state."""
    _patch_fast_pipeline(monkeypatch)
    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.1, 2.2, 3.3, 4.4, 5.5]

    model_a = MultiConformerRegressor(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out_a"), verbose=False, seed=1
    )
    model_b = MultiConformerRegressor(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out_b"), verbose=False, seed=2
    )
    model_a.train(smiles_train, y_train)
    model_b.train(smiles_train, y_train)

    train_a = pd.read_csv(tmp_path / "out_a" / "train.csv")
    train_b = pd.read_csv(tmp_path / "out_b" / "train.csv")
    assert list(train_a["SMILES"]) != list(train_b["SMILES"])


def test_predict_before_train_raises(tmp_path):
    model = MultiConformerRegressor(output_folder=str(tmp_path / "out"), verbose=False)
    try:
        model.predict(["CCO"])
        assert False, "predict should fail before train/load"
    except RuntimeError as e:
        assert "not trained" in str(e)


def test_save_load_and_predict_from_smiles(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch)

    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.1, 2.2, 3.3, 4.4, 5.5]

    model = MultiConformerRegressor(
        num_conf=2,
        hopt=False,
        num_cpu=1,
        output_folder=str(tmp_path / "train_out"),
        verbose=False,
        seed=42,
    )
    model.train(smiles_train, y_train)
    model_path = tmp_path / "model.pkl"
    model.save(model_path)

    loaded = MultiConformerRegressor.load(model_path, output_folder=str(tmp_path / "pred_out"))
    preds = loaded.predict(["CCF"])
    assert isinstance(preds, list)
    assert len(preds) == 1

    preds2 = loaded.predict(["CCF", "CCO"])
    assert isinstance(preds2, list)
    assert len(preds2) == 2


def test_predict_fallback_to_run_lazy_when_no_lazy_model(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch)
    model = MultiConformerRegressor(output_folder=str(tmp_path / "out"), num_conf=2, num_cpu=1, hopt=False, verbose=False)
    try:
        model.predict(["CCF"])
        assert False, "predict should fail when lazy model is not trained"
    except RuntimeError:
        assert True


def test_predict_raises_when_lazy_model_state_is_not_trained(tmp_path):
    class FakeLazyMIL:
        is_trained = False

    model = MultiConformerRegressor(output_folder=str(tmp_path / "out"), verbose=False)
    model.best_consensus = ["RDKitGEOM|Mock"]
    model._lazy_model = cast(Any, FakeLazyMIL())

    try:
        model.predict(["CCF"])
        assert False, "predict should fail when the stored LazyMIL artifact is not trained"
    except RuntimeError as e:
        assert "LazyMIL model is not trained" in str(e)


def test_predict_fallback_to_mean_when_consensus_predictor_missing(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch)
    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.1, 2.2, 3.3, 4.4, 5.5]
    model = MultiConformerRegressor(output_folder=str(tmp_path / "out"), num_conf=2, num_cpu=1, hopt=False, verbose=False)
    model.train(smiles_train, y_train)

    model._consensus_search = None
    preds = model.predict(["CCF"])
    assert isinstance(preds, list)
    assert len(preds) == 1


def test_predict_raises_when_consensus_columns_missing(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch)
    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.1, 2.2, 3.3, 4.4, 5.5]
    model = MultiConformerRegressor(output_folder=str(tmp_path / "out"), num_conf=2, num_cpu=1, hopt=False, verbose=False)
    model.train(smiles_train, y_train)

    model.best_consensus = ["Missing|Model"]
    try:
        model.predict(["CCF"])
        assert False, "predict should fail when consensus columns are missing"
    except ValueError as e:
        assert "missing model columns" in str(e)


def test_save_before_train_raises(tmp_path):
    model = MultiConformerRegressor(output_folder=str(tmp_path / "out"), verbose=False)
    try:
        model.save(tmp_path / "model.pkl")
        assert False, "save should fail before train/load"
    except RuntimeError as e:
        assert "not trained" in str(e)


def test_save_handles_unpicklable_consensus(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch)
    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.1, 2.2, 3.3, 4.4, 5.5]
    model = MultiConformerRegressor(output_folder=str(tmp_path / "out"), num_conf=2, num_cpu=1, hopt=False, verbose=False)
    model.train(smiles_train, y_train)

    model._consensus_search = UnpicklableConsensus()
    model_path = tmp_path / "model.pkl"
    model.save(model_path)

    loaded = MultiConformerRegressor.load(model_path)
    assert loaded._consensus_search is None


