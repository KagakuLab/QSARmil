from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from qsarmil.modelling.meta import MultiConformerEstimator


@click.command(name="predict")
@click.option(
    "--test-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV file; first column is SMILES.",
)
@click.option(
    "--model-folder",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder saved by `qsarmil train` (the --output-folder you gave it).",
)
@click.option(
    "--output-file",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the predictions CSV.",
)
@click.option(
    "--accelerator",
    type=click.Choice(["cpu", "gpu"]),
    default=None,
    help="Device to run inference on, overriding the model's training-time choice (e.g. predict on CPU "
    "for a model trained on GPU). Defaults to whatever it was trained with.",
)
@click.option("--verbose", is_flag=True, default=False, help="Print progress output.")
def predict_command(
    test_path: Path,
    model_folder: Path,
    output_file: Path,
    accelerator: str | None,
    verbose: bool,
) -> None:
    """Predict on new SMILES using a model saved by `qsarmil train`."""

    df = pd.read_csv(test_path)

    model: MultiConformerEstimator = MultiConformerEstimator.load(str(model_folder))
    model._lazy_model.verbose = verbose

    preds = model.predict(df.iloc[:, 0].tolist(), accelerator=accelerator)

    out_df = df.copy()
    out_df["prediction"] = preds
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_file, index=False)

    click.echo(f"\nPredictions saved to {output_file}")
