Descriptor calculation
========================

``DescriptorWrapper`` unifies the interface
between descriptor tools that don't otherwise share one — RDKit's 3D
descriptors (:mod:`qsarmil.descriptor.rdkit`) and MolFeat's calculators
(e.g. ``Pharmacophore3D``, ``USRDescriptors``) — so both can be dropped into
the same pipeline interchangeably. It takes any callable that computes a
descriptor vector for a single ``(mol, conformer_id)`` pair, and applies it
across a bag of conformers per molecule.

``run()`` also cleans up the resulting descriptor matrix: values above a
fixed extreme-value threshold are treated as invalid, and any descriptor
column that is invalid for at least one molecule in the batch is dropped
from all molecules, so the final feature matrix has no ``NaN``/extreme
values left in it.

Once descriptors are computed, ``BagMinMaxScaler`` (from ``milearn``) scales
each descriptor column to a common range — fit on the training bags only,
then applied to validation/test bags with the same fitted scaling.

Example: SMILES → conformers → descriptors
---------------------------------------------

.. code-block:: python

   from rdkit import Chem
   from qsarmil.conformer import RDKitConformerGenerator
   from qsarmil.descriptor.wrapper import DescriptorWrapper
   from qsarmil.descriptor.rdkit import RDKitWHIM
   from milearn.preprocessing import BagMinMaxScaler

   smiles = [
       "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
       "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
   ]
   mols = [Chem.MolFromSmiles(smi) for smi in smiles]

   conf_gen = RDKitConformerGenerator(num_conf=10, random_seed=42)
   conformers = conf_gen.run(mols)

   desc_calc = DescriptorWrapper(RDKitWHIM(), verbose=True)
   x = desc_calc.run(conformers)

   scaler = BagMinMaxScaler()
   scaler.fit(x)
   x_scaled = scaler.transform(x)

A MolFeat descriptor is used the same way — just swap the transformer:

.. code-block:: python

   from molfeat.calc import Pharmacophore3D

   desc_calc = DescriptorWrapper(Pharmacophore3D(factory="pmapper"))
   x = desc_calc.run(conformers)

Available RDKit descriptors
------------------------------

- ``RDKitGEOM`` — geometric descriptors (asphericity, eccentricity, PMI,
  radius of gyration, etc.)
- ``RDKitAUTOCORR`` — 3D autocorrelation
- ``RDKitRDF`` — radial distribution function
- ``RDKitMORSE`` — Morse descriptors
- ``RDKitWHIM`` — WHIM descriptors
- ``RDKitGETAWAY`` — GETAWAY descriptors

Failure handling
-----------------

If descriptor calculation fails for a given molecule's bag, it is wrapped
in a ``FailedMolecule`` sentinel rather than
raising, consistent with :doc:`conformer`.
