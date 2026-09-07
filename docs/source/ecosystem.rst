Ecosystem
==========

QSARmil is the top-level package that ties together two companion
libraries, each responsible for one part of the pipeline.

- **QSARmil** — the end-to-end molecular MIL pipeline: conformer
  generation, descriptor calculation, the command-line interface, and
  orchestrating the two packages below.
  `github.com/KagakuLab/QSARmil <https://github.com/KagakuLab/QSARmil>`_
- **milearn** (PyPI: ``mikit-learn``) — the multi-instance learning
  algorithms QSARmil trains on top of descriptors: the MLP wrappers and the
  six networks described in :doc:`mil`, along with their stepwise
  hyperparameter optimization.
  `github.com/KagakuLab/milearn <https://github.com/KagakuLab/milearn>`_
- **QSARcons** (PyPI: ``qsarcons``) — consensus model selection. Used to
  pick which trained models to combine into a final prediction; see
  :doc:`consensus`.
  `github.com/KagakuLab/QSARcons <https://github.com/KagakuLab/QSARcons>`_

Both ``milearn`` and ``QSARcons`` are installed automatically as QSARmil
dependencies — see :doc:`installation`.
