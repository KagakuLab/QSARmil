import os

import pandas as pd
from conftest import MockEstimator

import qsarmil.modelling.lazy as lazy_mod
import qsarmil.modelling.meta as meta_mod
from qsarmil.descriptor.rdkit import RDKitGEOM
from qsarmil.descriptor.wrapper import DescriptorWrapper
from qsarmil.modelling.meta import (
    MultiConformerClassifier,
    MultiConformerEstimator,
    MultiConformerRegressor,
)


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


class FakeGeneticSearchBadConsensus:
    """Returns a consensus referencing a model column that doesn't exist."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def run(self, x_val, true_val):
        return ["Missing|Model"]

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


def test_regressor_forces_continuous_task():
    """MultiConformerRegressor's whole reason to exist: its LazyMIL is built with task="continuous"
    directly, rather than inferring it from the target values (which could be ambiguous for a
    2-distinct-value numeric target)."""
    model = MultiConformerRegressor(output_folder=None)
    assert model._lazy_model.task == "continuous"
    assert model._lazy_model.ESTIMATORS is lazy_mod.REGRESSORS


def test_classifier_forces_binary_task():
    model = MultiConformerClassifier(output_folder=None)
    assert model._lazy_model.task == "binary"
    assert model._lazy_model.ESTIMATORS is lazy_mod.CLASSIFIERS


def test_init_default_creates_temp_dir():
    model = MultiConformerRegressor()
    assert os.path.isdir(model.output_folder)
    assert model._lazy_model.seed == 42


def test_train_predict_end_to_end_regression(monkeypatch, tmp_path, capsys):
    _patch_fast_pipeline(monkeypatch)

    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.1, 2.2, 3.3, 4.4, 5.5]
    smiles_test = ["CCF"]

    model = MultiConformerRegressor(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=True, seed=42
    )
    preds = model.train_predict(smiles_train, y_train, smiles_test)

    assert isinstance(preds, list)
    assert len(preds) == 1
    assert len(model.best_consensus) > 0

    captured = capsys.readouterr()
    assert "Step-5. Genetic model consensus search" in captured.out
    assert "Best genetic consensus" in captured.out

    assert (tmp_path / "out" / "train.csv").exists()
    assert (tmp_path / "out" / "val.csv").exists()
    assert (tmp_path / "out" / "test.csv").exists()


def test_train_predict_end_to_end_classification_quiet(monkeypatch, tmp_path, capsys):
    _patch_fast_pipeline(monkeypatch, classifier=True)

    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [0, 1, 0, 1, 0]
    smiles_test = ["CCF"]

    model = MultiConformerClassifier(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False, seed=99
    )
    preds = model.train_predict(smiles_train, y_train, smiles_test)

    assert isinstance(preds, list)
    assert len(preds) == 1
    assert capsys.readouterr().out == ""


def test_seeds_produce_different_train_val_splits(monkeypatch, tmp_path):
    """self.seed reaches train_test_split's random_state."""
    _patch_fast_pipeline(monkeypatch)
    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.1, 2.2, 3.3, 4.4, 5.5]
    smiles_test = ["CCF"]

    model_a = MultiConformerRegressor(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out_a"), verbose=False, seed=1
    )
    model_b = MultiConformerRegressor(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out_b"), verbose=False, seed=2
    )
    model_a.train_predict(smiles_train, y_train, smiles_test)
    model_b.train_predict(smiles_train, y_train, smiles_test)

    train_a = pd.read_csv(tmp_path / "out_a" / "train.csv")
    train_b = pd.read_csv(tmp_path / "out_b" / "train.csv")
    assert list(train_a["SMILES"]) != list(train_b["SMILES"])


def test_train_predict_raises_when_consensus_columns_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(meta_mod, "GeneticSearch", FakeGeneticSearchBadConsensus)
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", {"RDKitGEOM": lambda: DescriptorWrapper(RDKitGEOM(), verbose=False)})
    monkeypatch.setattr(lazy_mod, "REGRESSORS", {"Mock": MockEstimator(supports_hopt=False)})

    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.1, 2.2, 3.3, 4.4, 5.5]
    model = MultiConformerRegressor(output_folder=str(tmp_path / "out"), num_conf=2, num_cpu=1, hopt=False, verbose=False)

    try:
        model.train_predict(smiles_train, y_train, ["CCF"])
        assert False, "train_predict should fail when consensus columns are missing"
    except ValueError as e:
        assert "missing model columns" in str(e)
