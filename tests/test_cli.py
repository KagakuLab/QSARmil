import pandas as pd
import pytest
from click.testing import CliRunner
from conftest import MockEstimator

import qsarmil.cli as cli_mod
import qsarmil.cli.train_predict as cli_train_predict_mod
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


def test_train_predict_regression(monkeypatch, tmp_path, regression_csv):
    _patch_fast_pipeline(monkeypatch)
    runner = CliRunner()
    output_folder = tmp_path / "mcfm"
    output_file = tmp_path / "predictions.csv"

    result = runner.invoke(
        cli_mod.cli,
        [
            "train_predict",
            "--train-path",
            str(regression_csv),
            "--test-path",
            str(regression_csv),
            "--task-type",
            "regression",
            "--output-folder",
            str(output_folder),
            "--output-file",
            str(output_file),
            "--num-conf",
            "2",
            "--num-cpu",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Predictions saved to" in result.output
    assert (output_folder / "train.csv").exists()
    assert (output_folder / "val.csv").exists()

    out_df = pd.read_csv(output_file)
    assert "prediction" in out_df.columns
    assert len(out_df) == 5
    assert list(out_df["smiles"]) == ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl"]


def test_train_predict_classification(monkeypatch, tmp_path, classification_csv):
    _patch_fast_pipeline(monkeypatch, classifier=True)
    runner = CliRunner()
    output_folder = tmp_path / "mcfm"
    output_file = tmp_path / "predictions.csv"

    result = runner.invoke(
        cli_mod.cli,
        [
            "train_predict",
            "--train-path",
            str(classification_csv),
            "--test-path",
            str(classification_csv),
            "--task-type",
            "classification",
            "--output-folder",
            str(output_folder),
            "--output-file",
            str(output_file),
            "--num-conf",
            "2",
            "--num-cpu",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    out_df = pd.read_csv(output_file)
    assert "prediction" in out_df.columns
    assert len(out_df) == 5


def test_train_predict_hopt_accepts_bool_value(monkeypatch, tmp_path, regression_csv):
    _patch_fast_pipeline(monkeypatch)
    runner = CliRunner()
    output_folder = tmp_path / "mcfm"
    output_file = tmp_path / "predictions.csv"

    result = runner.invoke(
        cli_mod.cli,
        [
            "train_predict",
            "--train-path",
            str(regression_csv),
            "--test-path",
            str(regression_csv),
            "--task-type",
            "regression",
            "--output-folder",
            str(output_folder),
            "--output-file",
            str(output_file),
            "--num-conf",
            "2",
            "--num-cpu",
            "1",
            "--hopt",
            "False",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_file.exists()


def test_train_predict_missing_columns_raises(tmp_path, regression_csv):
    df = pd.DataFrame({"only_one_column": ["CCO"]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "train_predict",
            "--train-path",
            str(path),
            "--test-path",
            str(regression_csv),
            "--task-type",
            "regression",
            "--output-folder",
            str(tmp_path / "out"),
            "--output-file",
            str(tmp_path / "predictions.csv"),
        ],
    )
    assert result.exit_code != 0
    assert "needs at least 2 columns" in result.output


def test_train_predict_verbose_prints_progress(monkeypatch, tmp_path, regression_csv):
    _patch_fast_pipeline(monkeypatch)
    runner = CliRunner()
    output_folder = tmp_path / "mcfm"
    output_file = tmp_path / "predictions.csv"

    result = runner.invoke(
        cli_mod.cli,
        [
            "train_predict",
            "--train-path",
            str(regression_csv),
            "--test-path",
            str(regression_csv),
            "--task-type",
            "regression",
            "--output-folder",
            str(output_folder),
            "--output-file",
            str(output_file),
            "--num-conf",
            "2",
            "--num-cpu",
            "1",
            "--verbose",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Step-1" in result.output


def test_train_predict_quiet_by_default(monkeypatch, tmp_path, regression_csv):
    _patch_fast_pipeline(monkeypatch)
    runner = CliRunner()
    output_folder = tmp_path / "mcfm"
    output_file = tmp_path / "predictions.csv"

    result = runner.invoke(
        cli_mod.cli,
        [
            "train_predict",
            "--train-path",
            str(regression_csv),
            "--test-path",
            str(regression_csv),
            "--task-type",
            "regression",
            "--output-folder",
            str(output_folder),
            "--output-file",
            str(output_file),
            "--num-conf",
            "2",
            "--num-cpu",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Step-1" not in result.output
    assert "Predictions saved to" in result.output


def test_train_predict_accelerator_choice_is_forwarded(monkeypatch, tmp_path, regression_csv):
    _patch_fast_pipeline(monkeypatch)
    captured_kwargs = {}
    original_init = cli_train_predict_mod.MultiConformerRegressor.__init__

    def spy_init(self, *args, **kwargs):
        captured_kwargs.update(kwargs)
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(cli_train_predict_mod.MultiConformerRegressor, "__init__", spy_init)

    runner = CliRunner()
    output_folder = tmp_path / "mcfm"
    output_file = tmp_path / "predictions.csv"
    result = runner.invoke(
        cli_mod.cli,
        [
            "train_predict",
            "--train-path",
            str(regression_csv),
            "--test-path",
            str(regression_csv),
            "--task-type",
            "regression",
            "--output-folder",
            str(output_folder),
            "--output-file",
            str(output_file),
            "--num-conf",
            "2",
            "--num-cpu",
            "1",
            "--accelerator",
            "gpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured_kwargs["accelerator"] == "gpu"


def test_train_predict_accelerator_rejects_auto(tmp_path, regression_csv):
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "train_predict",
            "--train-path",
            str(regression_csv),
            "--test-path",
            str(regression_csv),
            "--task-type",
            "regression",
            "--output-folder",
            str(tmp_path / "mcfm"),
            "--output-file",
            str(tmp_path / "predictions.csv"),
            "--accelerator",
            "auto",
        ],
    )
    assert result.exit_code != 0
    assert "auto" in result.output.lower()


def test_cli_help_and_version():
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["--help"])
    assert result.exit_code == 0
    assert "train-predict" in result.output.replace("_", "-") or "train_predict" in result.output

    result = runner.invoke(cli_mod.cli, ["--version"])
    assert result.exit_code == 0
