QSARmil - molecular multi-instance machine learning
============================================================

**Multi-conformer modeling edition**

``QSARmil`` builds Quantitative Structure-Activity Relationship (QSAR) models by representing each molecule as a
bag of instances and applying multi-instance machine learning to it. In this edition, instances are molecular
**3D conformers**. Representing molecules as bags of fragments, tautomers, or mixture components is planned, but
not yet implemented - see below for what's ready today.

**Ready to use:**

- 3D conformer generation (RDKit ETKDG embedding + UFF optimization)
- 3D descriptor calculation (RDKit and MolFeat descriptors)
- Multi-instance learning models (via ``milearn``) and consensus model search (via ``qsarcons``)
- Regression and binary classification tasks are currently supported

**Under development:**

- Fragment-based modeling
- Tautomer-based modeling
- Mixture-based modeling

Installation
--------------------------

**Option 1 - one-line install from scratch with conda:**

.. code-block:: bash

    conda env create -f environment.yml && conda activate qsarmil

**Option 2 - install into an existing environment:**

.. code-block:: bash

    pip install qsarmil

Beginner usage
--------------------------

For a predefined pipeline that takes zero QSAR/MIL knowledge and zero setup: hand it your training data and get
predictions for new molecules back.

.. code-block:: python

     from qsarmil.modelling.meta import MultiConformerRegressor

     smiles_train = [
         "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
         "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
         "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
         "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1",
         "OC(=O)C(C)c1cccc(c1)C(=O)c1ccccc1",
     ]
     y_train = [5.2, 5.9, 6.1, 7.0, 5.6]

     # train model
     model = MultiConformerRegressor(num_conf=10, hopt=True, output_folder="mcf", verbose=True)
     model.train(smiles_train, y_train)

     # predict for new molecules
     y_pred = model.predict(["CC(C(=O)O)Oc1cccc(c1)-c1ccccc1", "CC(C(=O)O)c1ccc(cc1)-c1ccc(F)cc1"])

Use ``MultiConformerRegressor`` for continuous properties and ``MultiConformerClassifier`` for binary
classification. See the full walkthrough in
`01_Beginner_Easy_Start.ipynb <notebooks/01_Beginner_Easy_Start.ipynb>`_.

Professional usage
--------------------------

Modify or build your own modeling pipeline by combining ``QSARmil``'s individual modules (conformer generator,
descriptor calculator, multi-instance methods, etc.) directly. See
`02_Professional_Pipeline_Customization.ipynb <notebooks/02_Professional_Pipeline_Customization.ipynb>`_ for the
details.

Tutorials
--------------------------

- `01_Beginner_Easy_Start.ipynb <notebooks/01_Beginner_Easy_Start.ipynb>`_ - the predefined, zero-configuration pipeline.
- `02_Professional_Pipeline_Customization.ipynb <notebooks/02_Professional_Pipeline_Customization.ipynb>`_ - building a custom pipeline from individual modules.
- `03_Key_Instance_Detection.ipynb <notebooks/03_Key_Instance_Detection.ipynb>`_ - identifying which conformers drive a prediction.

Documentation
--------------------------

Full documentation is available (link coming soon).