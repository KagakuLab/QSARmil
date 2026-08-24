from __future__ import annotations

from typing import Union

from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Mol

from qsarmil.utils.logging import FailedConformer, FailedMolecule

RDLogger.DisableLog("rdApp.*")

MolOrFailed = Union[Mol, FailedMolecule, FailedConformer]


class RDKitFragmentGenerator:
    """Generate molecular fragments using RDKit BRICS decomposition."""

    def __init__(self, verbose: bool = True) -> None:
        """Initialize the RDKit fragment generator."""
        super().__init__()
        self.verbose = verbose

    def _generate_fragments(self, mol: MolOrFailed) -> list[Mol] | FailedMolecule | FailedConformer:
        """Generate fragments for a single molecule using BRICS
        decomposition."""

        if isinstance(mol, (FailedMolecule, FailedConformer)):
            print("Failed molecule")
            return mol
        try:
            frag_smiles_set = BRICS.BRICSDecompose(mol)
            frags = [Chem.MolFromSmiles(smi) for smi in frag_smiles_set if smi]
            frags = [f for f in frags if f is not None]
        except Exception as e:
            print(e)
            frags = [mol]

        return frags

    def run(self, list_of_mols: list[MolOrFailed]) -> list[list[Mol] | FailedMolecule | FailedConformer]:
        """Generate fragments for a list of molecules."""

        total = len(list_of_mols)

        results = []
        for i, mol in enumerate(list_of_mols, 1):
            results.append(self._generate_fragments(mol))
            if self.verbose:
                print(f"Generating fragments: {i}/{total}", end="\r", flush=True)

        if self.verbose:
            print(f"Generating fragments: {total}/{total}")

        return results
