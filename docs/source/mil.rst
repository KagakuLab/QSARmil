Multi-instance learning
====================================

.. note::

   The MIL algorithms themselves live in the companion package
   `milearn <https://github.com/KagakuLab/milearn>`_. Its own documentation
   is not published yet, so the methods QSARmil relies on are summarized
   here instead.

All methods below take a list of bags (one array of instance descriptors
per molecule) and a matching list of target values, and follow the same
``model.fit(x, y)`` / ``model.predict(x)`` interface. Each is available as
a ``...Regressor`` (continuous targets) or ``...Classifier`` (binary
targets), e.g. ``AdditiveAttentionNetworkRegressor`` /
``AdditiveAttentionNetworkClassifier``.

MLP wrappers
-------------

- ``BagWrapperMLPNetwork`` — pools each bag into a single vector first
  (mean/sum/max/lse), then applies an MLP to that pooled vector.
- ``InstanceWrapperMLPNetwork`` — applies an MLP to every instance
  individually (each instance inherits its bag's label during training),
  then pools the per-instance predictions (mean) into a bag prediction.

These two exist for benchmarking, not because they're expected to be the
best-performing option. Both are built on the exact same underlying PyTorch
training code (the same instance transformer + estimator layers, optimizer,
and training loop) as the networks below — only the pooling
strategy/placement differs. That makes it possible to isolate the effect of
*where and how pooling happens* from everything else. Wrapping a real
scikit-learn ``MLPRegressor``/``MLPClassifier`` instead would not give a
fair comparison, since scikit-learn's MLP implementation (initialization,
optimizer, training loop) differs from milearn's PyTorch one in ways
unrelated to the pooling strategy being studied.

Networks
---------

- ``BagNetwork`` — transforms instances, pools them into a bag embedding
  (mean/sum/max/lse), then scores the pooled embedding. No learned
  weighting between instances.
- ``InstanceNetwork`` — transforms and scores each instance individually,
  then pools the per-instance scores (mean/sum/max) into a bag prediction.
  No learned weighting between instances.
- ``AdditiveAttentionNetwork`` — learns a per-instance attention score via
  a small MLP, softmax-normalized across the bag, and combines instances
  into a bag embedding as a weighted sum.
- ``SelfAttentionNetwork`` — instances attend to each other via
  scaled dot-product self-attention (query/key/value projections), then are
  combined into a bag embedding.
- ``HopfieldAttentionNetwork`` — a single learned query vector attends to
  every instance in the bag (Hopfield-style associative attention).
- ``DynamicPoolingNetwork`` — combines instances via iterative dynamic
  routing (capsule-network-style), rather than a single attention pass.

Unlike the two networks above, the attention-based networks and
``DynamicPoolingNetwork`` produce genuine per-instance weights that can be
inspected after training — see :doc:`kid`.

Parameters
-----------

Common to every method above (set at initialization):

- ``hidden_layer_sizes`` (default ``(256, 128, 64)``) — sizes of the hidden
  layers of the instance transformer.
- ``max_epochs`` (default ``1000``) — maximum training epochs.
- ``batch_size`` (default ``128``) — training batch size.
- ``activation`` (default ``"gelu"``) — one of ``"relu"``, ``"gelu"``,
  ``"leakyrelu"``, ``"elu"``, ``"silu"``.
- ``learning_rate`` (default ``0.001``).
- ``early_stopping`` (default ``True``) — stop training when validation
  loss stops improving (patience of 10 epochs).
- ``weight_decay`` (default ``0.0``).
- ``instance_dropout`` (default ``0.0``) — probability of randomly dropping
  real instances from a bag during training (at least one instance per bag
  is always kept).
- ``accelerator`` (default ``"cpu"``) — ``"cpu"`` or ``"gpu"``.
- ``verbose`` (default ``False``) — print training progress.
- ``random_seed`` (default ``42``).
- ``num_workers`` (default ``0``) — dataloader worker processes.

Additional, method-specific:

- ``pool`` — ``BagNetwork``/``BagWrapperMLPNetwork`` support
  ``"mean"``/``"sum"``/``"max"``/``"lse"``; ``InstanceNetwork`` supports
  ``"mean"``/``"sum"``/``"max"``; ``InstanceWrapperMLPNetwork`` only
  supports ``"mean"``.
- ``tau`` (default ``1.0``) — attention temperature/scale, for
  ``AdditiveAttentionNetwork``, ``SelfAttentionNetwork``, and
  ``HopfieldAttentionNetwork`` only. Lower values sharpen the attention
  distribution.
