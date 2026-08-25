Multi-instance learning
====================================

QSARmil delegates the core MIL algorithms to the companion package
`milearn <https://github.com/KagakuLab/milearn>`_ (distributed on PyPI as
``mikit-learn``). This page documents which methods are exposed and when to
use each.

Available methods
-------------------

TODO: list the MIL algorithms exposed through QSARmil's estimator classes
(e.g. instance pooling strategies, attention-based pooling, dynamic pooling
networks, etc. — whatever milearn actually implements), with a short
one-line description of each.

Choosing a method
-------------------

TODO: practical guidance — which method to reach for by default, and when
to try alternatives (dataset size, interpretability needs, compute budget).

CPU / GPU
----------

TODO: cross-reference :doc:`installation` for the ``accelerator`` option and
note which methods benefit most from GPU acceleration (e.g. neural
network-based poolers vs. simple aggregation methods).
