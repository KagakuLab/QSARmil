import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qsarmil.utils.ensemble import ConformerEnsemble, FragmentEnsemble, MixtureEnsemble
from qsarmil.utils.logging import FailedConformer


def _embedded_mol(smiles="CCO", num_conf=3):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    return mol


def test_conformer_ensemble_splits_each_conformer():
    mol = _embedded_mol(num_conf=3)
    ensemble = ConformerEnsemble(mol)
    assert len(ensemble) == 3
    for conf_mol in ensemble:
        assert conf_mol.GetNumConformers() == 1


def test_conformer_ensemble_rejects_failed_conformer():
    with pytest.raises(ValueError):
        ConformerEnsemble(FailedConformer(None))


def test_conformer_ensemble_rejects_zero_conformers():
    mol = Chem.MolFromSmiles("CCO")  # never embedded, has 0 conformers
    with pytest.raises(ValueError):
        ConformerEnsemble(mol)


def test_fragment_ensemble_wraps_mols():
    mols = [Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CCC")]
    ensemble = FragmentEnsemble(mols)
    assert list(ensemble) == mols


def test_fragment_ensemble_empty_default():
    assert list(FragmentEnsemble()) == []


def test_mixture_ensemble_wraps_mols():
    mols = [Chem.MolFromSmiles("CC")]
    ensemble = MixtureEnsemble(mols)
    assert list(ensemble) == mols
