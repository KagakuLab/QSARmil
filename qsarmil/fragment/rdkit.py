from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS

from qsarmil.utils.ensemble import FragmentEnsemble
from qsarmil.utils.logging import FailedConformer, FailedMolecule

RDLogger.DisableLog("rdApp.*")


class RDKitFragmentGenerator:
    """Generate molecular fragments using RDKit BRICS decomposition."""

    def __init__(self, verbose=True):
        """Initialize the RDKit fragment generator."""
        super().__init__()
        self.verbose = verbose

    def _generate_fragments(self, mol):
        """Generate fragments for a single molecule using BRICS
        decomposition."""

        if isinstance(mol, (FailedMolecule, FailedConformer)):
            print("Failed molecule")
            return mol
        try:
            frag_smiles_set = BRICS.BRICSDecompose(mol)
            frags = [Chem.MolFromSmiles(smi) for smi in frag_smiles_set if smi]
            frags = FragmentEnsemble([f for f in frags if f is not None])
        except Exception as e:
            print(e)
            frags = FragmentEnsemble([mol])

        return frags

    def run(self, list_of_mols):
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
