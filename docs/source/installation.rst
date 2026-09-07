Installation
=============

QSARmil is built on three core packages: `RDKit <https://www.rdkit.org/>`_
(conformer generation and 3D descriptors), `MolFeat
<https://molfeat-docs.datamol.io/>`_ (additional 3D descriptors), and
`click <https://click.palletsprojects.com/>`_ (the command-line interface).

Two companion packages are installed automatically as dependencies and power
the modeling side of the pipeline:

- `milearn <https://github.com/KagakuLab/milearn>`_ (distributed on PyPI as
  ``mikit-learn``) — the multi-instance learning algorithms, see :doc:`mil`.
- `QSARcons <https://github.com/KagakuLab/QSARcons>`_ (distributed on PyPI as
  ``qsarcons``) — consensus model selection, see :doc:`consensus`.

You do not need to install either of them separately.

Option 1 — one-line install from scratch with conda
------------------------------------------------------

.. code-block:: bash

    conda env create -f environment.yml && conda activate qsarmil

Option 2 — install into an existing environment
---------------------------------------------------

.. code-block:: bash

    pip install qsarmil
