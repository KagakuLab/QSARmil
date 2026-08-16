import joblib
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from qsarmil.utils.ensemble import ConformerEnsemble
from qsarmil.utils.logging import FailedConformer, FailedMolecule

RDLogger.DisableLog("rdApp.*")


class ConformerGenerator:
    """Generate and optimize molecular conformers with optional filtering."""

    def __init__(self, num_conf=10, e_thresh=None, num_cpu=1, verbose=True):
        """Store the generation settings used by every run() call.

        Args:
            num_conf (int): Number of conformers to embed per molecule.
            e_thresh (float, optional): Energy cutoff for dropping high-energy
                conformers. If None, no energy filtering is applied.
            num_cpu (int): Number of threads to use when generating conformers
                in parallel.
            verbose (bool): Whether to print a progress indicator.
        """
        super().__init__()

        self.num_conf = num_conf
        self.e_thresh = e_thresh
        self.num_cpu = num_cpu
        self.verbose = verbose

    def _prepare_molecule(self, mol):
        """Prepare a molecule by adding explicit hydrogens."""
        mol = Chem.AddHs(mol)
        return mol

    def _embed_conformers(self, mol):
        """Generate multiple 3D conformers for a molecule."""
        mol = self._prepare_molecule(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMultipleConfs(mol, numConfs=self.num_conf, params=params)
        return mol

    def _generate_conformers(self, mol):
        """Generate and optionally filter conformers for a molecule."""

        if isinstance(mol, (FailedMolecule, FailedConformer)):
            return mol
        try:
            mol = self._embed_conformers(mol)
            if not mol.GetNumConformers():
                print(f"Conformer generation failed for {Chem.MolToSmiles(mol)}")
                return FailedConformer(mol)
            mol = self._optimize_conformers(mol)
        except Exception:
            return FailedConformer(mol)

        if self.e_thresh is not None:
            mol = filter_by_energy(mol, self.e_thresh)

        return mol

    def _optimize_conformers(self, mol):
        """Optimize all conformers of a molecule using UFF force field."""

        for conf in mol.GetConformers():
            AllChem.UFFOptimizeMolecule(mol, confId=conf.GetId())
        return mol

    def run(self, list_of_mols):
        """Generate conformers for a list of molecules in parallel."""

        total = len(list_of_mols)
        completed = [0]
        verbose = self.verbose

        class PrintCallback(joblib.parallel.BatchCompletionCallBack):
            """Joblib batch callback that prints a running progress count."""

            def __call__(self, *args, **kwargs):
                """Update the progress count and forward to the real callback."""
                completed[0] += self.batch_size
                if verbose:
                    print(f"Generating conformers: {min(completed[0], total)}/{total}", end="\r", flush=True)
                return super().__call__(*args, **kwargs)

        old_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = PrintCallback

        try:
            results = Parallel(n_jobs=self.num_cpu, backend="threading")(
                delayed(self._generate_conformers)(mol) for mol in list_of_mols
            )

            results = [ConformerEnsemble(i) for i in results]
        finally:
            joblib.parallel.BatchCompletionCallBack = old_callback

        if verbose:
            print(f"Generating conformers: {total}/{total}")

        return results


def filter_by_energy(mol, e_thresh=1):
    """Filter conformers of a molecule based on relative energy."""

    conf_energy_list = []
    for conf in mol.GetConformers():
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())
        if ff is None:
            continue
        conf_energy_list.append((conf.GetId(), ff.CalcEnergy()))
    conf_energy_list = sorted(conf_energy_list, key=lambda x: x[1])

    min_energy = conf_energy_list[0][1]
    for conf_id, conf_energy in conf_energy_list[1:]:
        if conf_energy - min_energy >= e_thresh:
            mol.RemoveConformer(conf_id)

    return mol
