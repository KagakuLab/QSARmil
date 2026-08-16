from __future__ import annotations

import logging
import os
import sys
import threading
from types import TracebackType
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import Mol

RDLogger.DisableLog("rdApp.*")


class FailedMolecule:
    """Represents a molecule for which SMILES parsing or initialization failed.

    Attributes:
        smiles (str): The SMILES string that failed to parse.
    """

    def __init__(self, smiles: str) -> None:
        """Initialize a FailedMolecule with the problematic SMILES.

        Args:
            smiles (str): SMILES string that failed parsing.
        """
        super().__init__()
        self.smiles = smiles

    def __str__(self) -> str:
        """Return a human-readable error message.

        Returns:
            str: Error message describing the parsing failure.
        """
        return f"{self.smiles} -> SMILES parsing failed"


class FailedConformer:
    """Represents a molecule for which conformer generation failed.

    Attributes:
        mol (rdkit.Chem.Mol | None): Molecule that failed conformer
            generation. Can be ``None`` when the input molecule itself was
            already invalid (e.g. an unparseable SMILES upstream).
    """

    def __init__(self, mol: Mol | None) -> None:
        """Initialize a FailedConformer with the failed molecule.

        Args:
            mol (rdkit.Chem.Mol | None): Molecule that failed conformer
                generation, or ``None`` if the input was already invalid.
        """
        super().__init__()
        self.mol = mol

    def __str__(self) -> str:
        """Return a human-readable error message.

        Returns:
            str: Error message describing the conformer generation failure.
        """
        smi = Chem.MolToSmiles(self.mol)
        return f"{smi} -> conformer generation failed"


class FailedDescriptor:
    """Represents a molecule (or conformer/fragment ensemble) for which
    descriptor calculation failed.

    Attributes:
        mol: The molecule or ensemble that failed descriptor calculation.
    """

    def __init__(self, mol: Any) -> None:
        """Initialize a FailedDescriptor with the failed input.

        Args:
            mol: The molecule or ensemble that failed descriptor calculation.
        """
        super().__init__()
        self.mol = mol

    def __str__(self) -> str:
        """Return a human-readable error message.

        Returns:
            str: Error message describing the descriptor calculation failure.
        """
        smi = Chem.MolToSmiles(self.mol)
        return f"{smi} -> descriptor calculation failed"


class OutputSuppressor:
    """Context manager that silences all output while it's open.

    Suppresses Python `print`/logging as well as C/C++ libraries writing
    straight to stdout/stderr (e.g. CatBoost, XGBoost). Thread-safe and
    nestable: nested or concurrent uses share one counter, so output only
    comes back once every ``with`` block has exited.
    """

    _lock = threading.Lock()
    _active = 0

    def __enter__(self) -> None:
        """Redirect stdout, stderr and logging to /dev/null for this thread."""
        with OutputSuppressor._lock:
            if OutputSuppressor._active == 0:
                # Save original file descriptors
                self._orig_stdout_fd = os.dup(1)
                self._orig_stderr_fd = os.dup(2)

                # Open null file
                self._devnull = os.open(os.devnull, os.O_WRONLY)

                # Redirect Python-level stdio
                self._orig_stdout = sys.stdout
                self._orig_stderr = sys.stderr
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")

                # Redirect C-level stdout/stderr
                os.dup2(self._devnull, 1)
                os.dup2(self._devnull, 2)

                # Disable logging
                logging.disable(logging.CRITICAL)

            OutputSuppressor._active += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Restore the original stdout, stderr and logging state."""
        with OutputSuppressor._lock:
            OutputSuppressor._active -= 1
            if OutputSuppressor._active == 0:
                # Restore file descriptors
                os.dup2(self._orig_stdout_fd, 1)
                os.dup2(self._orig_stderr_fd, 2)

                # Close temp files
                os.close(self._devnull)
                os.close(self._orig_stdout_fd)
                os.close(self._orig_stderr_fd)

                # Restore Python-level stdio
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = self._orig_stdout
                sys.stderr = self._orig_stderr

                # Re-enable logging
                logging.disable(logging.NOTSET)
