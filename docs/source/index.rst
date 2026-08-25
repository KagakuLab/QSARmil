QSARmil documentation
======================

**QSARmil** is a Python package for molecular multi-instance machine learning
(MIL) applied to QSAR (quantitative structure-activity relationship) modeling.

Instead of representing a molecule as a single feature vector, QSARmil treats
it as a *bag of instances* (e.g. conformers, fragments) and learns directly
from that structure — letting the model decide which instances matter most
for a given property.

.. code-block:: python

   from qsarmil import MultiConformerRegressor

   smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
   y = [3.5, 3.9]

   model = MultiConformerRegressor()
   model.fit(smiles, y)
   model.predict(smiles)

Companion packages `milearn <https://github.com/KagakuLab/milearn>`_ (MIL
algorithms) and `QSARcons <https://github.com/KagakuLab/QSARcons>`_
(consensus modeling) power parts of the pipeline under the hood — see
:doc:`ecosystem` for how the pieces fit together.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   review
   installation
   training_data
   conformer_generation
   descriptor_calculation
   mil_methods
   hyperparameter_optimization
   key_instance_detection
   consensus_modelling
   ecosystem
   cli
   tutorials
