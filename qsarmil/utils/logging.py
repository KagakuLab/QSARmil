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
    """Represents a molecule for which SMILES parsing or initialization failed."""

    def __init__(self, smiles: str) -> None:
        """Store the SMILES string that failed to parse."""
        super().__init__()
        self.smiles = smiles

    def __str__(self) -> str:
        """Return a human-readable error message."""
        return f"{self.smiles} -> SMILES parsing failed"


class FailedConformer:
    """Represents a molecule for which conformer generation failed."""

    def __init__(self, mol: Mol | None) -> None:
        """Store the molecule that failed conformer generation, or ``None`` if the input was already invalid."""
        super().__init__()
        self.mol = mol

    def __str__(self) -> str:
        """Return a human-readable error message."""
        smi = Chem.MolToSmiles(self.mol)
        return f"{smi} -> conformer generation failed"


class FailedDescriptor:
    """Represents a molecule or bag of conformers for which descriptor calculation failed."""

    def __init__(self, mol: Any) -> None:
        """Store the molecule or bag that failed descriptor calculation."""
        super().__init__()
        self.mol = mol

    def __str__(self) -> str:
        """Return a human-readable error message, using the first conformer's SMILES if given a whole bag."""
        mol = self.mol
        if isinstance(mol, list):
            mol = mol[0] if mol else None
        smi = Chem.MolToSmiles(mol) if mol is not None else "?"
        return f"{smi} -> descriptor calculation failed"


def print_step_header(step: int, title: str, bar_width: int = 26) -> None:
    """Print a ``+---+ / Step-N. Title / +---+`` banner for a pipeline stage.

    Args:
        step (int): Step number to display (e.g. ``1``).
        title (str): Short label for the step (e.g. ``"SMILES parsing"``).
        bar_width (int): Number of ``+`` characters in the banner rule.
    """

    bar = "+" * bar_width
    print(f"\n{bar}")
    print(f"Step-{step}. {title}")
    print(bar)


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
