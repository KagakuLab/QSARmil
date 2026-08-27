from __future__ import annotations

import os
from typing import Any

import joblib
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Mol

from qsarmil.utils.logging import FailedMolecule

RDLogger.DisableLog("rdApp.*")


def filter_by_energy(mol: Mol, e_thresh: float = 1) -> Mol:
    """Drop conformers whose UFF energy is more than e_thresh above the lowest-energy conformer."""

    conf_with_energy = []
    for conf in mol.GetConformers():
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())
        if ff is None:
            continue
        conf_with_energy.append((conf.GetId(), ff.CalcEnergy()))
    conf_with_energy = sorted(conf_with_energy, key=lambda x: x[1])

    min_energy = conf_with_energy[0][1]
    for conf_id, conf_energy in conf_with_energy[1:]:
        if conf_energy - min_energy >= e_thresh:
            mol.RemoveConformer(conf_id)

    return mol


class RDKitConformerGenerator:
    """Generate and optimize molecular conformers (RDKit ETKDG embedding + UFF optimization) with optional filtering."""

    def __init__(
        self,
        num_conf: int = 10,
        e_thresh: float | None = None,
        num_cpu: int = os.cpu_count() or 1,
        verbose: bool = True,
        random_seed: int = 42,
    ) -> None:
        """Store the generation settings used by every run() call."""
        super().__init__()

        self.num_conf = num_conf
        self.e_thresh = e_thresh
        self.num_cpu = num_cpu
        self.verbose = verbose
        self.random_seed = random_seed

    def _embed_conformers(self, mol: Mol) -> Mol:
        """Add hydrogens and embed multiple 3D conformers for a molecule."""
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed
        AllChem.EmbedMultipleConfs(mol, numConfs=self.num_conf, params=params)
        return mol

    def _optimize_conformers(self, mol: Mol) -> Mol:
        """Optimize all conformers of a molecule using the UFF force field."""

        for conf in mol.GetConformers():
            AllChem.UFFOptimizeMolecule(mol, confId=conf.GetId())
        return mol

    def _generate_conformers(self, mol: Mol | None | FailedMolecule) -> Mol | FailedMolecule:
        """Embed, optimize, and optionally filter conformers for one molecule; pass failed sentinels through."""

        if isinstance(mol, FailedMolecule):
            return mol
        if mol is None:
            return FailedMolecule(mol, message="SMILES parsing failed")

        embedded = self._embed_conformers(mol)
        if not embedded.GetNumConformers():
            print(f"Conformer generation failed for {Chem.MolToSmiles(embedded)}")
            return FailedMolecule(embedded, message="conformer generation failed")

        optimized = self._optimize_conformers(embedded)
        if self.e_thresh is not None:
            return filter_by_energy(optimized, self.e_thresh)
        return optimized

    def _split_into_conformers(self, mol: Mol) -> list[Mol]:
        """Split a multi-conformer molecule into a list of single-conformer copies."""

        conf_list = []
        for conf in mol.GetConformers():
            conf_mol = Chem.Mol(mol)
            conf_mol.RemoveAllConformers()
            conf_mol.AddConformer(conf, assignId=True)
            conf_list.append(conf_mol)
        return conf_list

    def run(self, list_of_mols: list[Mol | None | FailedMolecule]) -> list[list[Mol] | FailedMolecule]:
        """Generate conformers for a list of molecules in parallel; failures pass through instead of raising."""

        total = len(list_of_mols)
        completed = [0]
        verbose = self.verbose

        class PrintCallback(joblib.parallel.BatchCompletionCallBack):
            """Joblib batch callback that prints a running progress count."""

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
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
            results = [mol if isinstance(mol, FailedMolecule) else self._split_into_conformers(mol) for mol in results]
        finally:
            joblib.parallel.BatchCompletionCallBack = old_callback

        if verbose:
            print(f"Generating conformers: {total}/{total}")

        return results
