Conformer generation
======================

``RDKitConformerGenerator`` is a thin wrapper
around RDKit's conformer embedding and optimization: for each input
molecule it embeds several 3D conformers (ETKDG), optimizes them (UFF), and
returns each molecule as a *bag* of single-conformer molecules — the
multi-instance representation used everywhere downstream.

Example
--------

.. code-block:: python

   from rdkit import Chem
   from qsarmil.conformer import RDKitConformerGenerator

   smiles = [
       "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
       "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
   ]
   mols = [Chem.MolFromSmiles(smi) for smi in smiles]

   conf_gen = RDKitConformerGenerator(num_conf=10, num_cpu=2, verbose=True, random_seed=42)
   conformers = conf_gen.run(mols)

``conformers`` is a list with one entry per input molecule: either a list of
single-conformer ``Mol`` objects (the bag), or a
``FailedMolecule`` sentinel if that molecule
could not be embedded.

Parameters
-----------

- ``num_conf`` (default ``10``) — number of conformers to embed per molecule
  (RDKit's ``ETKDGv3``).
- ``e_thresh`` (default ``None``) — if set, conformers whose UFF energy is
  more than ``e_thresh`` above the lowest-energy conformer for that molecule
  are dropped after optimization. Leave as ``None`` to keep every embedded
  conformer.
- ``num_cpu`` (default: all available CPUs) — number of threads used to
  generate conformers in parallel.
- ``verbose`` (default ``True``) — print a running "generated X/Y" progress
  count while conformers are being generated.
- ``random_seed`` (default ``42``) — seed for the ETKDG embedding, so
  results are reproducible.

Failure handling
-----------------

Molecules that fail to parse or fail to embed any conformer are wrapped in
a ``FailedMolecule`` sentinel rather than
raising an exception, so a handful of problematic molecules doesn't stop
the rest of the batch from running.
