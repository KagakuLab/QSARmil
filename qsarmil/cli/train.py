from __future__ import annotations

import os
from pathlib import Path

import click
import pandas as pd

from qsarmil.modelling.meta import MultiConformerClassifier, MultiConformerRegressor

TASK_CLASSES = {"regression": MultiConformerRegressor, "classification": MultiConformerClassifier}


@click.command(name="train")
@click.option(
    "--train-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV file; first column is SMILES, second column is the target property.",
)
@click.option(
    "--task-type",
    required=True,
    type=click.Choice(list(TASK_CLASSES)),
    help="Whether the target is a continuous property or a binary class label.",
)
@click.option(
    "--output-folder",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for the trained model and intermediate files - this is also what you pass to `qsarmil "
    "predict --model-folder` later.",
)
@click.option("--num-conf", default=10, show_default=True, help="Number of conformers to generate per molecule.")
@click.option("--hopt", type=bool, default=False, show_default=True, help="Hyperparameter-tune each estimator.")
@click.option(
    "--num-cpu", default=os.cpu_count() or 1, show_default=True, help="Number of CPU threads for conformer generation."
)
@click.option(
    "--accelerator",
    type=click.Choice(["cpu", "gpu"]),
    default="cpu",
    show_default=True,
    help="Training device - an explicit choice, never auto-detected.",
)
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--verbose", is_flag=True, default=False, help="Print progress output.")
def train_command(
    train_path: Path,
    task_type: str,
    output_folder: Path,
    num_conf: int,
    hopt: bool,
    num_cpu: int,
    accelerator: str,
    seed: int,
    verbose: bool,
) -> None:
    """Train a MultiConformerRegressor/Classifier on a labeled SMILES dataset."""

    df = pd.read_csv(train_path)
    if df.shape[1] < 2:
        raise click.UsageError(f"{train_path} needs at least 2 columns (SMILES, target); found {df.shape[1]}.")

    smiles = df.iloc[:, 0].tolist()
    y = df.iloc[:, 1].astype(float).tolist() if task_type == "regression" else df.iloc[:, 1].tolist()

    model = TASK_CLASSES[task_type](
        num_conf=num_conf,
        hopt=hopt,
        num_cpu=num_cpu,
        output_folder=str(output_folder),
        verbose=verbose,
        seed=seed,
        accelerator=accelerator,
    )
    model.train(smiles, y)
    model.save()

    click.echo(f"\nModel saved to {output_folder}")
