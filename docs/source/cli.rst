Interfaces (CLI)
===================

QSARmil ships a ``click``-based command-line interface, installed as the
``qsarmil`` command, with a single subcommand: ``train_predict``.

.. code-block:: bash

   qsarmil train_predict \
       --train-path train.csv \
       --test-path test.csv \
       --task-type regression \
       --output-folder ./results \
       --num-conf 10 \
       --hopt True \
       --verbose

Both ``--train-path`` and ``--test-path`` point to CSV files: the first
column is SMILES, and for the training file the second column is the
target property. ``--task-type`` is either ``regression`` or
``classification``.

Options
--------

- ``--train-path`` (required) — training CSV; first column SMILES, second
  column target.
- ``--test-path`` (required) — test CSV; first column SMILES.
- ``--task-type`` (required) — ``regression`` or ``classification``.
- ``--output-folder`` — where model files (``train.csv``/``val.csv``/
  ``test.csv``) are written. Defaults to a timestamped folder if omitted.
- ``--num-conf`` (default ``10``) — conformers generated per molecule.
- ``--hopt`` (default ``False``) — enable hyperparameter optimization for
  each estimator (see :doc:`hyperparameter`).
- ``--num-cpu`` (default: all available CPUs) — threads used for conformer
  generation.
- ``--accelerator`` (default ``"cpu"``) — ``cpu`` or ``gpu``.
- ``--random-seed`` (default ``42``).
- ``--verbose`` — print progress output as the pipeline runs.

Predictions are written to ``test.csv`` inside the output folder.
