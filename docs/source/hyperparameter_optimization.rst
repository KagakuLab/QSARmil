Hyperparameter optimization
==============================

QSARmil includes a built-in hyperparameter optimization step, exposed via
the ``hopt`` argument.

.. code-block:: python

   TODO: minimal Python example with hopt=True

Via the CLI:

.. code-block:: bash

   qsarmil train data.csv --hopt True

TODO: document the search strategy used (e.g. grid/random/Bayesian), the
default search space per model type, and how many trials are run by
default.

``LazyMIL``
------------

TODO: document ``LazyMIL.run()`` — it accepts a single ``(smiles, y)`` pair
and handles the train/validation split internally (``val_size`` is fixed
and not user-configurable). Explain when to reach for ``LazyMIL`` versus
instantiating an estimator directly.
