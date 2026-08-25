Training data
=============

QSARmil takes plain SMILES strings and a matching list of target property
values as input — there is no required DataFrame schema or column-naming
convention.

.. code-block:: python

   smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
   y = [3.5, 3.9]

TODO: document expected input shapes for classification vs. regression
targets, handling of missing/invalid SMILES, and how many molecules are
recommended as a practical minimum.

Handling problematic input
---------------------------

QSARmil uses a **failure-sentinel pattern** rather than raising exceptions
when a molecule, conformer, or descriptor fails to compute. Invalid entries
are wrapped in ``FailedMolecule``, ``FailedConformer``, or
``FailedDescriptor`` sentinels and reported rather than silently dropped or
crashing the whole pipeline.

TODO: short example showing a bad SMILES going in and the resulting
sentinel/verbose report coming out.

Train / validation split
-------------------------

TODO: document that ``val_size`` is handled internally (not user-facing)
and briefly explain why the split is hardcoded rather than exposed.
