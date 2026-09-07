Hyperparameter optimization
==============================

MIL networks and MLP wrappers (see :doc:`mil`) optimize their
hyperparameters **stepwise**: one hyperparameter at a time, in a fixed
order, keeping every other hyperparameter at its best value found so far.
This is cheaper than a full grid or random search over the joint parameter
space, at the cost of not exploring parameter interactions.

Availability
-------------

Stepwise optimization is ready for ``BagWrapperMLPNetwork`` /
``InstanceWrapperMLPNetwork`` and all six networks listed in :doc:`mil`.

It is **not** implemented yet for wrappers around classical estimators
(e.g. a scikit-learn ``RandomForestRegressor`` wrapped via milearn's
generic ``BagWrapper``/``InstanceWrapper``) — calling ``.hopt()`` on those
raises ``NotImplementedError``.

Usage
------

.. code-block:: python

   from milearn.network.regressor import AdditiveAttentionNetworkRegressor

   model = AdditiveAttentionNetworkRegressor(accelerator="cpu")
   model.hopt(x_train_scaled, y_train, verbose=True)
   model.fit(x_train_scaled, y_train)

Fixing a parameter
--------------------

Any parameter you pass at initialization that is **not** part of the
hyperparameter grid stays fixed at that value throughout the search — for
example ``accelerator``, ``random_seed``, ``verbose``, or ``num_workers``,
none of which are searched.

A parameter that *is* part of the grid (e.g. ``activation``,
``learning_rate``) is always optimized over the grid's options during
``.hopt()``, regardless of what you passed at initialization. To fix one of
those to a specific value instead of searching it, pass a custom
``param_grid`` with that entry set to a single (non-list) value rather than
a list of options — see below.

Default parameter grid
-------------------------

Calling ``.hopt(x, y)`` without a ``param_grid`` uses milearn's built-in
default:

.. code-block:: python

   DEFAULT_PARAM_GRID = {
       # Fixed hparams
       "max_epochs": 1000,
       "early_stopping": True,
       "accelerator": "cpu",
       "random_seed": 42,
       "verbose": False,

       # Activation function
       "activation": ["relu", "leakyrelu", "gelu", "elu", "silu"],

       # Learning dynamics
       "learning_rate": [1e-4, 1e-3],
       "batch_size": [32, 512, 1024],
       "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],

       # MIL specific
       "tau": [0.01, 0.5, 1.0],
       "instance_dropout": [0.0, 0.2, 0.4, 0.6, 0.8],
       "pool": ["mean", "sum", "max", "lse"],

       # Architecture depth/shape
       "hidden_layer_sizes": [(2048, 1024, 512, 256, 128, 64), (256, 128, 64), (128,)],
   }

Entries given as a single value (e.g. ``"max_epochs": 1000``) are treated
as fixed; entries given as a list are searched. Parameters that don't apply
to a given method (e.g. ``pool`` for an attention network) are silently
ignored.

.. note::

   QSARmil's own beginner pipeline (``LazyMIL``,
   used when ``hopt=True``) uses a narrower internal grid — only
   ``hidden_layer_sizes``, ``activation``, and ``learning_rate`` are
   searched, with ``max_epochs``, ``early_stopping``, ``accelerator``,
   ``random_seed``, and ``verbose`` fixed. This keeps the lazy, 72-model
   sweep tractable; use the network classes directly (as above) for full
   control over the search space.

Custom parameter grid
------------------------

.. code-block:: python

   custom_grid = {
       "max_epochs": 1000,
       "early_stopping": True,
       "accelerator": "cpu",
       "random_seed": 42,
       "verbose": False,
       "activation": ["relu", "gelu"],
       "learning_rate": [1e-4, 1e-3],
       "hidden_layer_sizes": [(256, 128, 64)],
   }
   model.hopt(x_train_scaled, y_train, param_grid=custom_grid, verbose=True)
