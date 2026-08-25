Installation
============

Conda (recommended)
--------------------

.. code-block:: bash

   TODO: conda one-liner from the README

pip
---

.. code-block:: bash

   pip install qsarmil

.. note::

   The multi-instance learning backend is provided by ``milearn``, published
   on PyPI under the distribution name ``mikit-learn``. This is pulled in
   automatically as a dependency — you should not need to install it
   separately.

GPU support
-----------

TODO: document the ``accelerator="gpu"`` / ``accelerator="cpu"`` option,
including the note that plain ``pip install torch==2.6.0`` resolves to a
CUDA 12.4 build by default, and how to override the accelerator at
inference time independently of how the model was trained.

Requirements
------------

TODO: Python version support, RDKit version, OS notes.
