from __future__ import annotations

import os
from pathlib import Path

import click
import pandas as pd

from qsarmil.modelling.meta import MultiConformerClassifier, MultiConformerRegressor

TASK_CLASSES = {"regression": MultiConformerRegressor, "classification": MultiConformerClassifier}


@click.group()
@click.version_option(package_name="qsarmil")
def cli() -> None:
    """QSARmil: train and apply multi-conformer multi-instance learning models on SMILES data."""


@cli.command(name="train_predict")
@click.option(
    "--train-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV file; first column is SMILES, second column is the target property.",
)
@click.option(
    "--test-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV file; first column is SMILES.",
)
@click.option(
    "--task-type",
    required=True,
    type=click.Choice(list(TASK_CLASSES)),
    help="Whether the target is a continuous property or a binary class label.",
)
@click.option(
    "--output-folder",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for the model's files (train.csv/val.csv/test.csv). Defaults to a timestamped folder.",
)
@click.option("--num-conf", default=10, show_default=True, help="Number of conformers to generate per molecule.")
@click.option("--hopt", type=bool, default=False, show_default=True, help="Hyperparameter optimization for each estimator.")
@click.option(
    "--num-cpu", default=os.cpu_count() or 1, show_default=True, help="Number of CPU threads for conformer generation."
)
@click.option(
    "--accelerator", type=click.Choice(["cpu", "gpu"]), default="cpu", show_default=True, help="Training device."
)
@click.option("--random-seed", default=42, show_default=True, help="Random seed.")
@click.option("--verbose", is_flag=True, default=False, help="Print progress output.")
def train_predict_command(
    train_path: Path,
    test_path: Path,
    task_type: str,
    output_folder: Path | None,
    num_conf: int,
    hopt: bool,
    num_cpu: int,
    accelerator: str,
    random_seed: int,
    verbose: bool,
) -> None:
    """Train a MultiConformerRegressor/Classifier and predict on new SMILES, in one run."""

    if accelerator == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    train_df = pd.read_csv(train_path)
    if train_df.shape[1] < 2:
        raise click.UsageError(f"{train_path} needs at least 2 columns (SMILES, target); found {train_df.shape[1]}.")

    test_df = pd.read_csv(test_path)

    smiles_train = train_df.iloc[:, 0].tolist()
    y_train = (
        train_df.iloc[:, 1].astype(float).tolist() if task_type == "regression" else train_df.iloc[:, 1].tolist()
    )

    model = TASK_CLASSES[task_type](
        num_conf=num_conf,
        hopt=hopt,
        num_cpu=num_cpu,
        output_folder=str(output_folder) if output_folder is not None else None,
        verbose=verbose,
        random_seed=random_seed,
        accelerator=accelerator,
    )
    model.train_predict(smiles_train, y_train, test_df.iloc[:, 0].tolist())

    click.echo(f"\nPredictions saved to {os.path.join(model.output_folder, 'test.csv')}")


if __name__ == "__main__":
    cli()
