from __future__ import annotations

import click

from qsarmil.cli.train_predict import train_predict_command


@click.group()
@click.version_option(package_name="qsarmil")
def cli() -> None:
    """QSARmil: train and apply multi-conformer multi-instance learning models on SMILES data."""


cli.add_command(train_predict_command)


if __name__ == "__main__":
    cli()
