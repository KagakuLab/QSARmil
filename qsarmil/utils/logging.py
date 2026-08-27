from __future__ import annotations

import logging
import os
import sys
import threading
from types import TracebackType
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


class FailedMolecule:
    """A single sentinel for anything that failed anywhere in the pipeline (parsing, conformers, descriptors)."""

    def __init__(self, mol: Any, message: str = "failed") -> None:
        """Store the failed item - a SMILES string, a Mol, or a bag of Mols - plus why it failed."""
        super().__init__()
        self.mol = mol
        self.message = message

    def __str__(self) -> str:
        """Return a human-readable "<SMILES> -> <reason>" message."""
        mol = self.mol
        if isinstance(mol, list):
            mol = mol[0] if mol else None
        if isinstance(mol, str):
            smi = mol
        elif mol is not None:
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                smi = "?"
        else:
            smi = "?"
        return f"{smi} -> {self.message}"


class OutputSuppressor:
    """Context manager that silences all stdout/stderr/logging output while it's open, nestably and thread-safely."""

    _lock = threading.Lock()
    _active = 0

    def __enter__(self) -> None:
        """Redirect stdout/stderr and logging to /dev/null for this thread."""
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
