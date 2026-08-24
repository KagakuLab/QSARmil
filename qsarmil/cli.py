from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from qsarmil.modelling.meta import MultiConformerClassifier, MultiConformerEstimator, MultiConformerRegressor

TASK_CLASSES = {"regression": MultiConformerRegressor, "classification": MultiConformerClassifier}


@click.group()
@click.version_option(package_name="qsarmil")
def cli() -> None:
    """QSARmil: train and apply multi-conformer multi-instance learning models on SMILES data."""


@cli.command()
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
    help="Directory for the trained model and intermediate files.",
)
@click.option(
    "--model-path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to save the trained model [default: <output-folder>/model.pkl].",
)
@click.option("--num-conf", default=10, show_default=True, help="Number of conformers to generate per molecule.")
@click.option("--hopt", type=bool, default=False, show_default=True, help="Hyperparameter-tune each estimator.")
@click.option("--num-cpu", default=20, show_default=True, help="Number of CPU threads for conformer generation.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--verbose", is_flag=True, default=False, help="Print progress output.")
def train(
    train_path: Path,
    task_type: str,
    output_folder: Path,
    model_path: Path | None,
    num_conf: int,
    hopt: bool,
    num_cpu: int,
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
    )
    model.train(smiles, y)

    model_path = model_path or (output_folder / "model.pkl")
    model.save(model_path)

    click.echo(f"\nModel saved to {model_path}")


@cli.command()
@click.option(
    "--test-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV file; first column is SMILES.",
)
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a model saved by `qsarmil train`.",
)
@click.option(
    "--output-file",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the predictions CSV.",
)
@click.option("--verbose", is_flag=True, default=False, help="Print progress output.")
def predict(
    test_path: Path,
    model_path: Path,
    output_file: Path,
    verbose: bool,
) -> None:
    """Predict on new SMILES using a model saved by `qsarmil train`."""

    df = pd.read_csv(test_path)

    model: MultiConformerEstimator = MultiConformerEstimator.load(model_path)
    model.verbose = verbose
    if model._lazy_model is not None:
        model._lazy_model.verbose = verbose

    preds = model.predict(df.iloc[:, 0].tolist())

    out_df = df.copy()
    out_df["prediction"] = preds
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_file, index=False)

    click.echo(f"\nPredictions saved to {output_file}")


if __name__ == "__main__":
    cli()
