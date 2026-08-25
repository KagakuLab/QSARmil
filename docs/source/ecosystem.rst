Ecosystem
=========

QSARmil is the top-level package that ties together two companion
libraries, each responsible for one part of the pipeline.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Package
     - Role
     - Link
   * - **QSARmil**
     - End-to-end molecular MIL pipeline for QSAR: conformers, descriptors,
       CLI, consensus orchestration.
     - `github.com/KagakuLab/qsarmil <https://github.com/KagakuLab/qsarmil>`_
   * - **milearn**
     - Multi-instance learning algorithms (pooling strategies, dynamic
       pooling networks, etc.). Published on PyPI as ``mikit-learn``.
     - `github.com/KagakuLab/milearn <https://github.com/KagakuLab/milearn>`_
   * - **QSARcons**
     - Consensus modeling — combining predictions from multiple models.
     - `github.com/KagakuLab/QSARcons <https://github.com/KagakuLab/QSARcons>`_

Why split into three packages?
---------------------------------

TODO: short rationale — e.g. milearn is general-purpose MIL and useful
outside of cheminformatics; QSARcons is general-purpose consensus modeling
and useful outside of MIL; QSARmil is the molecule-specific glue on top.

Version compatibility
-----------------------

TODO: document any version pinning requirements between the three packages
(e.g. QSARmil X.Y requires milearn >= A.B).
