Descriptor calculation
=======================

Each conformer/instance is converted into a numeric feature vector via a
``DescriptorWrapper``.

.. code-block:: python

   TODO: minimal example calling DescriptorWrapper.run()

Available descriptor types
----------------------------

TODO: list supported descriptor families (e.g. 3D descriptors, fingerprints,
physicochemical properties — whatever is actually implemented).

Post-processing
-----------------

``DescriptorWrapper.postprocess()`` handles NaN columns and reports which
columns were dropped and why (verbose mode). This runs automatically as
part of ``.run()`` unless disabled.

TODO: example showing the verbose column-drop report output.

Failure handling
-----------------

Descriptors that cannot be computed for a given instance are wrapped in a
``FailedDescriptor`` sentinel, consistent with the failure-sentinel pattern
used throughout the pipeline (see :doc:`training_data`).

Train-time dropping vs. predict-time imputation
--------------------------------------------------

TODO: explain the separation of concerns — columns/rows dropped at train
time are handled differently from missing values encountered at predict
time (imputed rather than dropped, so predictions are never silently
skipped).
