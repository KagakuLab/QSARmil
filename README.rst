
QSARmil - molecular multi-instance machine learning
============================================================
``QSARmil`` is a package for designing pipelines for building QSAR models with multi-instance machine learning algorithms.

Introduction
--------------------------
Multi-instance machine learning for molecules

Installation
--------------------------

.. code-block:: bash

    pip install qsarmil

Benchmarking QSARmil
--------------------------

To facilitate benchmarking ``QSARmil`` against alternative platforms, we developed a meta-model builder that eliminates
the need for manual adjustments to the model-building protocol. The pipeline automatically generates multiple
multi-conformer models using diverse descriptor sets and multi-instance learning methods, and then applies
a genetic algorithm to identify the optimal consensus combination of individual models. The input data should be provided
as a list of molecular SMILES strings and a list of the corresponding target property values.
Use ``MultiConformerRegressor`` for continuous properties and ``MultiConformerClassifier`` for binary classification.

.. code-block:: python

     from qsarmil.modelling.meta import MultiConformerRegressor

     model = MultiConformerRegressor(num_conf=10, hopt=True, output_folder="mcf", verbose=True)
     model.train(smiles_train, y_train)
     y_pred = model.predict(smiles_test)

Use cases
--------------------------

See the examples of ``QSARmil`` application for different tasks in the `tutorial collection <notebooks>`_ .

Command-line interface
--------------------------

``QSARmil`` also installs a ``qsarmil`` command for training and predicting without writing any Python:

.. code-block:: bash

    qsarmil train --train-path regression.csv --task-type regression --output-folder mcfm
    qsarmil predict --test-path regression.csv --model-path mcfm/model.pkl --output-file mcfm/test_predictions.csv

``--train-path``/``--test-path`` point to a CSV whose first column is SMILES and, for training, whose second column
is the target property. ``--task-type`` is ``regression`` or ``classification``. Run ``qsarmil train --help`` or
``qsarmil predict --help`` for the full list of options (``--num-conf``, ``--hopt``, ``--num-cpu``, ``--seed``,
``--verbose``, and more).

Development
--------------------------

To set up a development environment, clone the repository and install it in
editable mode along with the ``dev`` dependency group (``ruff``, ``mypy``,
``pytest``, ``pytest-cov``):

.. code-block:: bash

    git clone https://github.com/KagakuLab/QSARmil.git
    cd QSARmil
    pip install -e . --group dev

``--group`` requires pip 25.1 or newer (``pip install --upgrade pip`` if needed).

Run the test suite (coverage is reported automatically):

.. code-block:: bash

    pytest

Lint and type-check:

.. code-block:: bash

    ruff check .
    mypy qsarmil