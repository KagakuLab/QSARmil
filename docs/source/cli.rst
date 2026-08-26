Command-line interface (CLI)
==============================

QSARmil ships a ``click``-based command-line interface with a single command, ``train_predict``, which trains
a model and predicts on a test set in one run. There's no separate save/load step - training and prediction
happen in the same process.

``train_predict``
-------------------

.. code-block:: bash

   qsarmil train_predict --train-path train.csv --test-path test.csv --task-type regression \
       --output-folder ./results --output-file ./results/predictions.csv --hopt True --verbose

Input CSV format: first column SMILES, second column (``--train-path`` only) the target value - no header-name
flags, the columns are read by position.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``--train-path``
     - required
     - CSV with training data (SMILES + target).
   * - ``--test-path``
     - required
     - CSV with SMILES to predict on.
   * - ``--task-type``
     - required
     - ``regression`` or ``classification``.
   * - ``--output-folder``
     - required
     - Where intermediate files (``train.csv``/``val.csv``/``test.csv``) are written.
   * - ``--output-file``
     - required
     - Where the predictions CSV is written.
   * - ``--hopt``
     - ``False``
     - Explicit ``True``/``False``. Enables hyperparameter optimization
       (see :doc:`hyperparameter_optimization`).
   * - ``--accelerator``
     - ``cpu``
     - ``cpu`` or ``gpu`` - an explicit choice, never auto-detected.
   * - ``--verbose``
     - ``False``
     - Quiet by default; enables the five-step progress printing with
       per-step timing/memory.

TODO: fill in the remaining options (``--num-conf``, ``--num-cpu``, ``--seed``) and defaults directly from
``qsarmil/cli/train_predict.py`` - the table above is a starting point, not exhaustive.

Progress output
-----------------

When ``--verbose`` is set, each of the five pipeline steps prints a header plus timing/memory usage for that
step.

TODO: paste a real example of the verbose console output here.
