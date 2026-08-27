from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Mol

from qsarmil.utils.logging import FailedMolecule

RDLogger.DisableLog("rdApp.*")


class RDKitFragmentGenerator:
    """Generate molecular fragments using RDKit BRICS decomposition."""

    def __init__(self, verbose: bool = True) -> None:
        """Store the verbosity setting."""
        super().__init__()
        self.verbose = verbose

    def _transform(self, mol: Mol | FailedMolecule) -> list[Mol] | FailedMolecule:
        """Generate fragments for one molecule using BRICS decomposition; pass failed sentinels through."""

        if isinstance(mol, FailedMolecule):
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

    def run(self, list_of_mols: list[Mol | FailedMolecule]) -> list[list[Mol] | FailedMolecule]:
        """Generate fragments for a list of molecules."""

        total = len(list_of_mols)

        results = []
        for i, mol in enumerate(list_of_mols, 1):
            results.append(self._transform(mol))
            if self.verbose:
                print(f"Generating fragments: {i}/{total}", end="\r", flush=True)

        return results
