import os

import pandas as pd
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


def _patch_fast_pipeline(monkeypatch, classifier=False):
    monkeypatch.setattr(meta_mod, "GeneticSearch", FakeGeneticSearch)
    monkeypatch.setattr(lazy_mod, "DESCRIPTORS", {"RDKitGEOM": DescriptorWrapper(RDKitGEOM(), verbose=False)})
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
