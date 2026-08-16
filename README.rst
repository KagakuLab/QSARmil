
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
as a pandas DataFrame, where the first column contains the molecular SMILES strings and the second column contains
the corresponding target property values.

.. code-block:: python

     from qsarmil.meta import MultiConformerModel

     model = MultiConformerModel(num_conf=10, hopt=True, output_folder="mcf", verbose=True)
     y_pred = model.run_predict(df_train, df_test)

Use cases
--------------------------

See the examples of ``QSARmil`` application for different tasks in the `tutorial collection <notebooks>`_ .

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