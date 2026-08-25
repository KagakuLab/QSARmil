Interfaces (CLI)
==================

QSARmil ships a ``click``-based command-line interface with two
subcommands: ``train`` and ``predict``.

``train``
----------

.. code-block:: bash

   qsarmil train data.csv --output-folder ./results --hopt True --verbose

Input CSV format: positional argument, first column SMILES, second column
target value (no header-name flags — the columns are read by position).

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``--output-folder``
     - TODO
     - Where trained model artifacts are written.
   * - ``--hopt``
     - ``False``
     - Explicit ``True``/``False``. Enables hyperparameter optimization
       (see :doc:`hyperparameter_optimization`).
   * - ``--verbose``
     - ``False``
     - Quiet by default; enables the five-step progress printing with
       per-step timing/memory.

``predict``
------------

.. code-block:: bash

   qsarmil predict data.csv --model-path ./results/model.pkl

TODO: fill in the actual full option list and defaults for both
subcommands directly from ``qsarmil/cli.py`` — the table above is a
starting point, not exhaustive.

Progress output
-----------------

When ``--verbose`` is set, each of the five pipeline steps prints a header
plus timing/memory usage for that step.

TODO: paste a real example of the verbose console output here.
