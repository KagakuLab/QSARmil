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
    assert model._lazy_model.seed == 42
    assert model._lazy_model.accelerator == "cpu"


def test_init_passes_accelerator_through_unvalidated(tmp_path):
    """The library no longer validates `accelerator` itself - any value is stored as given
    and forwarded to the underlying estimators; CLI-level Choice validation covers end users."""
    model = MultiConformerRegressor(output_folder=str(tmp_path / "out"), accelerator="auto")
    assert model._lazy_model.accelerator == "auto"


def test_train_threads_accelerator_into_lazy_model(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch)
    smiles_train = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]
    y_train = [1.1, 2.2, 3.3, 4.4, 5.5]

    model = MultiConformerRegressor(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False, accelerator="gpu"
    )
    model.train(smiles_train, y_train)

    assert model._lazy_model.accelerator == "gpu"


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
        assert False, "predict should fail before train"
    except RuntimeError as e:
        assert "not trained" in str(e)


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
