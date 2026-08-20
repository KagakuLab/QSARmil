import os
import pickle

import pandas as pd
import pytest
from conftest import MockEstimator

import qsarmil.lazy as lazy_mod
import qsarmil.meta as meta_mod
from qsarmil.descriptor.rdkit import RDKitGEOM
from qsarmil.descriptor.wrapper import DescriptorWrapper
from qsarmil.meta import MultiConformerModel


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


def test_init_default_creates_temp_dir():
    model = MultiConformerModel()
    assert os.path.isdir(model.output_folder)
    assert model.seed == 42


def test_run_predict_fills_missing_test_target(monkeypatch, tmp_path, capsys):
    _patch_fast_pipeline(monkeypatch)

    df_train = pd.DataFrame({0: ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"], 1: [1.1, 2.2, 3.3, 4.4, 5.5]})
    df_test = pd.DataFrame({0: ["CCF"]})  # single column -> fake target gets filled in

    model = MultiConformerModel(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=True, seed=42
    )
    pred_df = model.run_predict(df_train, df_test)

    assert list(pred_df.columns) == ["SMILES", "pred"]
    assert len(pred_df) == 1

    captured = capsys.readouterr()
    assert "Running genetic consensus search" in captured.out
    assert "isn't controlled by `seed`" in captured.out
    assert "Best consensus" in captured.out


def test_train_then_predict_split_api(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch)

    df_train = pd.DataFrame({0: ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"], 1: [1.1, 2.2, 3.3, 4.4, 5.5]})
    df_test = pd.DataFrame({0: ["CCF"]})

    model = MultiConformerModel(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False, seed=42
    )
    model.train(df_train)

    assert model.is_trained is True
    assert len(model.best_consensus) > 0

    pred_df = model.predict(df_test)
    assert list(pred_df.columns) == ["SMILES", "pred"]
    assert len(pred_df) == 1


def test_run_predict_with_existing_test_target_quiet(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch, classifier=True)

    df_train = pd.DataFrame({0: ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"], 1: [0, 1, 0, 1, 0]})
    df_test = pd.DataFrame({0: ["CCF"], 1: [1]})  # already has a target column

    model = MultiConformerModel(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out"), verbose=False, seed=99
    )
    pred_df = model.run_predict(df_train, df_test)

    assert list(pred_df.columns) == ["SMILES", "pred"]
    assert len(pred_df) == 1


def test_seeds_produce_different_train_val_splits(monkeypatch, tmp_path):
    """self.seed reaches train_test_split's random_state."""
    _patch_fast_pipeline(monkeypatch)
    df_train = pd.DataFrame({0: ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"], 1: [1.1, 2.2, 3.3, 4.4, 5.5]})
    df_test = pd.DataFrame({0: ["CCF"], 1: [0.5]})

    model_a = MultiConformerModel(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out_a"), verbose=False, seed=1
    )
    model_b = MultiConformerModel(
        num_conf=2, hopt=False, num_cpu=1, output_folder=str(tmp_path / "out_b"), verbose=False, seed=2
    )
    model_a.run_predict(df_train.copy(), df_test.copy())
    model_b.run_predict(df_train.copy(), df_test.copy())

    train_a = pd.read_csv(tmp_path / "out_a" / "train.csv")
    train_b = pd.read_csv(tmp_path / "out_b" / "train.csv")
    assert list(train_a["SMILES"]) != list(train_b["SMILES"])


def test_predict_before_train_raises(tmp_path):
    model = MultiConformerModel(output_folder=str(tmp_path / "out"), verbose=False)
    df_test = pd.DataFrame({0: ["CCO"]})
    try:
        model.predict(df_test)
        assert False, "predict should fail before train/load"
    except RuntimeError as e:
        assert "not trained" in str(e)


def test_save_load_and_predict_from_smiles(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch)

    df_train = pd.DataFrame({0: ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"], 1: [1.1, 2.2, 3.3, 4.4, 5.5]})

    model = MultiConformerModel(
        num_conf=2,
        hopt=False,
        num_cpu=1,
        output_folder=str(tmp_path / "train_out"),
        verbose=False,
        seed=42,
    )
    model.train(df_train)
    model_path = tmp_path / "model.pkl"
    model.save(model_path)

    loaded = MultiConformerModel.load(model_path, output_folder=str(tmp_path / "pred_out"))
    pred_df = loaded.predict(pd.DataFrame({0: ["CCF"]}))
    assert list(pred_df.columns) == ["SMILES", "pred"]
    assert len(pred_df) == 1

    pred_df2 = MultiConformerModel.predictFromSMILES(model_path, ["CCF", "CCO"])
    assert list(pred_df2.columns) == ["SMILES", "pred"]
    assert len(pred_df2) == 2


def test_predict_fallback_to_run_lazy_when_no_lazy_model(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch)
    df_train = pd.DataFrame({0: ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"], 1: [1.1, 2.2, 3.3, 4.4, 5.5]})
    model = MultiConformerModel(output_folder=str(tmp_path / "out"), num_conf=2, num_cpu=1, hopt=False, verbose=False)
    model.train(df_train)

    model._lazy_model = None
    pred_df = model.predict(pd.DataFrame({0: ["CCF"]}))
    assert list(pred_df.columns) == ["SMILES", "pred"]


def test_predict_fallback_to_mean_when_consensus_predictor_missing(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch)
    df_train = pd.DataFrame({0: ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"], 1: [1.1, 2.2, 3.3, 4.4, 5.5]})
    model = MultiConformerModel(output_folder=str(tmp_path / "out"), num_conf=2, num_cpu=1, hopt=False, verbose=False)
    model.train(df_train)

    model._consensus_search = None
    pred_df = model.predict(pd.DataFrame({0: ["CCF"]}))
    assert list(pred_df.columns) == ["SMILES", "pred"]
    assert len(pred_df) == 1


def test_predict_raises_when_consensus_columns_missing(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch)
    df_train = pd.DataFrame({0: ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"], 1: [1.1, 2.2, 3.3, 4.4, 5.5]})
    model = MultiConformerModel(output_folder=str(tmp_path / "out"), num_conf=2, num_cpu=1, hopt=False, verbose=False)
    model.train(df_train)

    model.best_consensus = ["Missing|Model"]
    try:
        model.predict(pd.DataFrame({0: ["CCF"]}))
        assert False, "predict should fail when consensus columns are missing"
    except ValueError as e:
        assert "missing model columns" in str(e)


def test_save_before_train_raises(tmp_path):
    model = MultiConformerModel(output_folder=str(tmp_path / "out"), verbose=False)
    try:
        model.save(tmp_path / "model.pkl")
        assert False, "save should fail before train/load"
    except RuntimeError as e:
        assert "not trained" in str(e)


def test_save_handles_unpicklable_consensus(monkeypatch, tmp_path):
    _patch_fast_pipeline(monkeypatch)
    df_train = pd.DataFrame({0: ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"], 1: [1.1, 2.2, 3.3, 4.4, 5.5]})
    model = MultiConformerModel(output_folder=str(tmp_path / "out"), num_conf=2, num_cpu=1, hopt=False, verbose=False)
    model.train(df_train)

    model._consensus_search = UnpicklableConsensus()
    model_path = tmp_path / "model.pkl"
    model.save(model_path)

    loaded = MultiConformerModel.load(model_path)
    assert loaded._consensus_search is None


def test_load_raises_on_missing_metadata(monkeypatch, tmp_path):
    """Exercise the raise FileNotFoundError path in MultiConformerModel.load()."""
    _patch_fast_pipeline(monkeypatch)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        MultiConformerModel.load(empty_dir)



