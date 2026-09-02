QSARmil documentation
======================

**QSARmil** is a Python package for molecular multi-instance machine learning
(MIL) applied to QSAR (quantitative structure-activity relationship) modeling.

Instead of representing a molecule as a single feature vector, QSARmil treats
it as a *bag of instances* (e.g. conformers, fragments) and learns directly
from that structure — letting the model decide which instances matter most
for a given property.

Companion packages `milearn <https://github.com/KagakuLab/milearn>`_ (MIL
algorithms) and `QSARcons <https://github.com/KagakuLab/QSARcons>`_
(consensus modeling) power parts of the pipeline under the hood — see
:doc:`ecosystem` for how the pieces fit together.

Review
------

A quick status overview of what is ready to use today versus what is still
under active development.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Feature
     - Status
     - Notes
   * - ``MultiConformerRegressor`` / ``MultiConformerClassifier``
     - TODO
     - TODO
   * - Conformer generation
     - TODO
     - TODO
   * - Descriptor calculation
     - TODO
     - TODO
   * - Command-line interface (``train`` / ``predict``)
     - TODO
     - TODO
   * - Key instance detection
     - TODO
     - TODO
   * - Consensus modelling
     - TODO
     - TODO

.. note::

   Keep this table current — update it alongside each release rather than
   letting it drift from the codebase.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   installation
   data
   conformer
   descriptor
   mil
   hyperparameter
   kid
   consensus
   ecosystem
   cli
   tutorials