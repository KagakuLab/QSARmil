import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qsarmil.descriptor.rdkit import RDKitGEOM
from qsarmil.descriptor.wrapper import DescriptorWrapper
from qsarmil.utils.ensemble import ConformerEnsemble, FragmentEnsemble, MixtureEnsemble
from qsarmil.utils.logging import FailedDescriptor


def _conformer_ensemble(smiles="CCO", num_conf=3):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    for conf in mol.GetConformers():
        AllChem.UFFOptimizeMolecule(mol, confId=conf.GetId())
    return ConformerEnsemble(mol)


def test_conformer_ensemble_branch():
    ensemble = _conformer_ensemble(num_conf=3)
    wrapper = DescriptorWrapper(RDKitGEOM(), verbose=False)
    bag = wrapper(ensemble)
    assert bag.shape == (3, 11)


def test_fragment_ensemble_branch():
    mols = [Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CCC")]
    ensemble = FragmentEnsemble(mols)
    wrapper = DescriptorWrapper(lambda mol, **kw: np.array([1.0, 2.0]), verbose=False)
    bag = wrapper(ensemble)
    assert bag.shape == (2, 2)


def test_mixture_ensemble_branch():
    mols = [Chem.MolFromSmiles("CC")]
    ensemble = MixtureEnsemble(mols)
    wrapper = DescriptorWrapper(lambda mol, **kw: np.array([1.0, 2.0, 3.0]), verbose=False)
    bag = wrapper(ensemble)
    assert bag.shape == (1, 3)


def test_unsupported_type_raises_type_error():
    wrapper = DescriptorWrapper(lambda mol, **kw: np.array([1.0]), verbose=False)
    with pytest.raises(TypeError):
        wrapper._ensemble_to_descriptors([Chem.MolFromSmiles("CC")])


def test_transform_catches_exceptions_and_returns_failed_descriptor(capsys):
    def broken_transformer(mol, **kw):
        raise ValueError("boom")

    wrapper = DescriptorWrapper(broken_transformer, verbose=False)
    ensemble = FragmentEnsemble([Chem.MolFromSmiles("CC")])
    result = wrapper(ensemble)
    assert isinstance(result, FailedDescriptor)
    captured = capsys.readouterr()
    assert "boom" in captured.out


def test_run_over_list_reports_progress(capsys):
    mols = [Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CCC")]
    ensembles = [FragmentEnsemble([m]) for m in mols]
    wrapper = DescriptorWrapper(lambda mol, **kw: np.array([1.0]), verbose=True)
    results = wrapper.run(ensembles)
    assert len(results) == 2
    captured = capsys.readouterr()
    assert "Calculating descriptors:" in captured.out


def test_run_quiet():
    ensembles = [FragmentEnsemble([Chem.MolFromSmiles("CC")])]
    wrapper = DescriptorWrapper(lambda mol, **kw: np.array([1.0]), verbose=False)
    results = wrapper.run(ensembles)
    assert len(results) == 1
