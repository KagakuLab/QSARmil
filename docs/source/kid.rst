Key instance detection
=========================

Some multi-instance models don't just predict a property — they also learn
*how much each instance contributed* to that prediction. In a multi-
conformer model, this means the model can point at the specific conformer
it thinks is responsible for the molecule's activity (often called the
**active conformer**). Finding that conformer is what QSARmil calls **key
instance detection (KID)**.

There are two general families of approach for deriving instance
importance: **model-agnostic** methods (e.g. perturbing or removing
instances and observing the effect on the prediction, independent of the
model's internals) and **model-based** methods (reading the importance
directly out of a model that was built to produce it). QSARmil uses the
model-based approach: the attention-based networks and
``DynamicPoolingNetwork`` (see :doc:`mil`) learn per-instance weights as
part of training, and expose them directly.

Usage
------

.. code-block:: python

   from milearn.network.regressor import DynamicPoolingNetworkRegressor

   model = DynamicPoolingNetworkRegressor(accelerator="cpu")
   model.fit(x_train_scaled, y_train)

   weights = model.get_instance_weights(x_test_scaled)

``weights`` is a list with one entry per molecule; each entry is an array
with one weight per conformer in that molecule's bag (padding already
removed). A higher weight means the model considered that conformer more
important for the prediction — the conformer with the highest weight is
the model's guess at the active conformer.

``get_instance_weights()`` is only available on methods with a real,
learned weighting mechanism: ``AdditiveAttentionNetwork``,
``SelfAttentionNetwork``, ``HopfieldAttentionNetwork``, and
``DynamicPoolingNetwork`` (and their ``...Regressor``/``...Classifier``
variants). ``BagNetwork``, ``InstanceNetwork``, and the two MLP wrappers
pool instances with a fixed strategy (mean/max/etc.) rather than a learned
one, so they have no per-instance weights to return.
