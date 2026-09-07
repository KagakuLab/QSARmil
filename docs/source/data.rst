Training data
==============

QSARmil expects two plain Python lists as input:

- **SMILES** — a list of molecule strings.
- **Target property** — a list of matching values, one per molecule: a
  continuous number for regression, or a binary (0/1) label for
  classification.

.. code-block:: python

   smiles = [
       "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
       "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
       "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
   ]
   y = [5.2, 5.9, 6.1]

There is no required DataFrame schema or column-naming convention — SMILES
and targets are passed as separate lists, in the same order.

Molecules that cannot be parsed or embedded into conformers are not dropped
silently: see :doc:`conformer` for how these are handled.

Beyond SMILES and targets
----------------------------

The lists above are all that the beginner pipeline
(``MultiConformerEstimator`` and its
subclasses) needs. If you build a custom pipeline instead (see
:doc:`conformer`, :doc:`descriptor`, and :doc:`mil`), you are not limited to
QSARmil's own conformer generator or descriptor set — you can supply your
own pre-computed conformers or descriptors directly to the multi-instance
learning models, as long as they are shaped as one bag (a list of
instances) per molecule.
