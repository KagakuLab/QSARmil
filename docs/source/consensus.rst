Consensus modelling
======================

Rather than picking one descriptor/method combination up front, QSARmil's
beginner pipeline trains many of them and then searches for the best-
performing *subset* to combine into a consensus prediction — averaging
predictions across models for regression, or majority-voting for
classification. This tends to be more robust than trusting any single
model's choice of descriptor and architecture.

This happens in two steps: ``LazyMIL`` trains
every descriptor/estimator combination and collects their predictions;
``qsarcons.consensus.GeneticSearch`` then searches over which combination
of those models to combine.

Step 1 — LazyMIL
------------------

``LazyMIL`` trains every combination of the 9 built-in descriptors and the
8 built-in estimators (see :doc:`mil`) on a train/validation split, and
predicts on train, validation, and test sets with each one.

.. code-block:: python

   from qsarmil.modelling.lazy import LazyMIL

   lazy_model = LazyMIL(
       task="continuous",  # or "binary" for classification
       num_conf=10,
       hopt=True,
       output_folder="results",
       verbose=True,
   )
   result_train, result_val, result_test = lazy_model.run(
       smiles_train, y_train, smiles_val, y_val, smiles_test
   )

Each returned DataFrame has one row per molecule and one column per
descriptor/estimator combination (named ``"<descriptor>|<estimator>"``),
plus a ``SMILES`` column and, for train/val, a ``Y_TRUE`` column.

Step 2 — GeneticSearch
------------------------

``GeneticSearch`` treats each trained model as a candidate and searches for
the subset that gives the best consensus score, evaluated on the
validation set:

.. code-block:: python

   from qsarcons.consensus import GeneticSearch

   x_val, y_val_true = result_val.iloc[:, 2:], result_val.iloc[:, 1]

   cons_search = GeneticSearch(cons_size="auto", n_iter=50)
   best_models = cons_search.run(x_val, y_val_true)

   x_test = result_test.iloc[:, 1:]
   y_test_pred = cons_search.predict(x_test[best_models])

- ``cons_size`` — number of models in the final consensus. ``"auto"``
  tries several candidate sizes (2, 4, 6, ... up to 14 by default) and
  keeps whichever performs best on the validation set; pass an integer to
  fix the size instead.
- ``n_iter`` — number of genetic algorithm generations used to search for
  the best-scoring subset at a given size.
- ``.predict()`` combines the selected models' predictions by averaging
  (regression, ``r2`` score used internally) or majority vote
  (classification, ``f1_score`` used internally).

This whole two-step flow (LazyMIL → GeneticSearch) is what
``MultiConformerEstimator.train_predict()`` runs end-to-end for the
beginner pipeline — see :doc:`tutorials`.
