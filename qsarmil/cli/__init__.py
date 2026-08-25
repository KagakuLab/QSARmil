from __future__ import annotations

import click

from qsarmil.cli.predict import predict_command
from qsarmil.cli.train import train_command


@click.group()
@click.version_option(package_name="qsarmil")
def cli() -> None:
    """QSARmil: train and apply multi-conformer multi-instance learning models on SMILES data."""


cli.add_command(train_command)
cli.add_command(predict_command)


if __name__ == "__main__":
    cli()
