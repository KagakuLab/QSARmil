QSARmil documentation
======================

**QSARmil** is a Python package for molecular multi-instance machine learning
(MIL) applied to QSAR (quantitative structure-activity relationship) modeling.

Instead of representing a molecule as a single feature vector, QSARmil treats
it as a *bag of instances* (e.g. conformers, fragments) and learns directly
from that structure — letting the model decide which instances matter most
for a given property.

QSARmil offers two levels of access:

- **Beginner** — a predefined, zero-configuration pipeline. Hand it SMILES
  and target values, get predictions back. See :doc:`installation` to get
  set up, then :doc:`tutorials` for the full walkthrough.
- **Professional** — build a custom pipeline from QSARmil's individual
  modules (conformer generation, descriptor calculation, MIL methods,
  hyperparameter optimization, consensus modeling) directly. Each of these
  is documented in its own page below.

Companion packages `milearn <https://github.com/KagakuLab/milearn>`_ (MIL
algorithms) and `QSARcons <https://github.com/KagakuLab/QSARcons>`_
(consensus modeling) power parts of the pipeline under the hood — see
:doc:`ecosystem` for how the pieces fit together.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   installation
   data
   conformer
   descriptor
   mil
   hyperparameter
   consensus
   kid
   ecosystem
   cli
   tutorials
