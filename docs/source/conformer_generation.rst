Conformer generation
=====================

Each molecule is expanded into a *bag* of 3D conformers — a plain
``list[Mol]`` — which forms the multi-instance representation used
downstream by the MIL models.

.. code-block:: python

   TODO: minimal example calling the conformer generation step directly

Configuration
--------------

TODO: document the available parameters (number of conformers, embedding
method, energy pruning/RMSD pruning if applicable, random seed).

Failure handling
-----------------

If conformer generation fails for a given molecule (e.g. embedding fails
in RDKit), the molecule is wrapped in a ``FailedConformer`` sentinel rather
than raising — the pipeline continues and the failure is reported.

TODO: example of a verbose failure report.
