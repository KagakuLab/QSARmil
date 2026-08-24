import pandas as pd
import pytest
from click.testing import CliRunner
from conftest import MockEstimator

import qsarmil.cli as cli_mod
import qsarmil.modelling.lazy as lazy_mod
import qsarmil.modelling.meta as meta_mod
from qsarmil.descriptor.rdkit import RDKitGEOM
from qsarmil.descriptor.wrapper import DescriptorWrapper


class FakeGeneticSearch:
    """Stand-in for qsarcons.consensus.GeneticSearch, which is too slow to run in tests."""

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


@pytest.fixture
def regression_csv(tmp_path):
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"],
            "y": [1.1, 2.2, 3.3, 4.4, 5.5],
        }
    )
    path = tmp_path / "regression.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def classification_csv(tmp_path):
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"],
            "y": [0, 1, 0, 1, 0],
        }
    )
    path = tmp_path / "classification.csv"
    df.to_csv(path, index=False)
    return path


def test_train_then_predict_regression(monkeypatch, tmp_path, regression_csv):
    _patch_fast_pipeline(monkeypatch)
    runner = CliRunner()
    output_folder = tmp_path / "mcfm"

    result = runner.invoke(
        cli_mod.cli,
        [
            "train",
            "--train-path",
            str(regression_csv),
            "--task-type",
            "regression",
            "--output-folder",
            str(output_folder),
            "--num-conf",
            "2",
            "--num-cpu",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    model_path = output_folder / "model.pkl"
    assert model_path.exists()
    assert "Model saved to" in result.output

    output_file = tmp_path / "predictions.csv"
    result = runner.invoke(
        cli_mod.cli,
        [
            "predict",
            "--test-path",
            str(regression_csv),
            "--model-path",
            str(model_path),
            "--output-file",
            str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_file.exists()
    assert "Predictions saved to" in result.output

    out_df = pd.read_csv(output_file)
    assert "prediction" in out_df.columns
    assert len(out_df) == 5
    assert list(out_df["smiles"]) == ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]


def test_train_then_predict_classification(monkeypatch, tmp_path, classification_csv):
    _patch_fast_pipeline(monkeypatch, classifier=True)
    runner = CliRunner()
    output_folder = tmp_path / "mcfm"

    result = runner.invoke(
        cli_mod.cli,
        [
            "train",
            "--train-path",
            str(classification_csv),
            "--task-type",
            "classification",
            "--output-folder",
            str(output_folder),
            "--num-conf",
            "2",
            "--num-cpu",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    model_path = output_folder / "model.pkl"
    assert model_path.exists()

    output_file = tmp_path / "predictions.csv"
    result = runner.invoke(
        cli_mod.cli,
        [
            "predict",
            "--test-path",
            str(classification_csv),
            "--model-path",
            str(model_path),
            "--output-file",
            str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output
    out_df = pd.read_csv(output_file)
    assert "prediction" in out_df.columns
    assert len(out_df) == 5


def test_train_custom_model_path(monkeypatch, tmp_path, regression_csv):
    _patch_fast_pipeline(monkeypatch)
    runner = CliRunner()
    output_folder = tmp_path / "mcfm"
    custom_model_path = tmp_path / "custom.pickle"

    result = runner.invoke(
        cli_mod.cli,
        [
            "train",
            "--train-path",
            str(regression_csv),
            "--task-type",
            "regression",
            "--output-folder",
            str(output_folder),
            "--model-path",
            str(custom_model_path),
            "--num-conf",
            "2",
            "--num-cpu",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert custom_model_path.exists()
    assert not (output_folder / "model.pkl").exists()


def test_train_hopt_accepts_bool_value(monkeypatch, tmp_path, regression_csv):
    _patch_fast_pipeline(monkeypatch)
    runner = CliRunner()
    output_folder = tmp_path / "mcfm"

    result = runner.invoke(
        cli_mod.cli,
        [
            "train",
            "--train-path",
            str(regression_csv),
            "--task-type",
            "regression",
            "--output-folder",
            str(output_folder),
            "--num-conf",
            "2",
            "--num-cpu",
            "1",
            "--hopt",
            "False",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_folder / "model.pkl").exists()


def test_train_missing_columns_raises(tmp_path):
    df = pd.DataFrame({"only_one_column": ["CCO"]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "train",
            "--train-path",
            str(path),
            "--task-type",
            "regression",
            "--output-folder",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "needs at least 2 columns" in result.output


def test_train_verbose_prints_progress(monkeypatch, tmp_path, regression_csv):
    _patch_fast_pipeline(monkeypatch)
    runner = CliRunner()
    output_folder = tmp_path / "mcfm"

    result = runner.invoke(
        cli_mod.cli,
        [
            "train",
            "--train-path",
            str(regression_csv),
            "--task-type",
            "regression",
            "--output-folder",
            str(output_folder),
            "--num-conf",
            "2",
            "--num-cpu",
            "1",
            "--verbose",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Step-1" in result.output


def test_train_quiet_by_default(monkeypatch, tmp_path, regression_csv):
    _patch_fast_pipeline(monkeypatch)
    runner = CliRunner()
    output_folder = tmp_path / "mcfm"

    result = runner.invoke(
        cli_mod.cli,
        [
            "train",
            "--train-path",
            str(regression_csv),
            "--task-type",
            "regression",
            "--output-folder",
            str(output_folder),
            "--num-conf",
            "2",
            "--num-cpu",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Step-1" not in result.output
    assert "Model saved to" in result.output


def test_cli_help_and_version():
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.output
    assert "predict" in result.output

    result = runner.invoke(cli_mod.cli, ["--version"])
    assert result.exit_code == 0
