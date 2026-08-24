import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qsarmil.descriptor.rdkit import RDKitGEOM
from qsarmil.descriptor.wrapper import DescriptorWrapper
from qsarmil.utils.logging import FailedDescriptor


def _conformer_bag(smiles="CCO", num_conf=3):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    for conf in mol.GetConformers():
        AllChem.UFFOptimizeMolecule(mol, confId=conf.GetId())

    bag = []
    for conf in mol.GetConformers():
        conf_mol = Chem.Mol(mol)
        conf_mol.RemoveAllConformers()
        conf_mol.AddConformer(conf, assignId=True)
        bag.append(conf_mol)
    return bag


def test_conformer_bag_descriptors():
    bag = _conformer_bag(num_conf=3)
    wrapper = DescriptorWrapper(RDKitGEOM(), verbose=False)
    result = wrapper(bag)
    assert result.shape == (3, 11)


def test_plain_mol_list_descriptors():
    """DescriptorWrapper works on a plain list[Mol] - no wrapper type needed."""
    mols = [Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CCC")]
    wrapper = DescriptorWrapper(lambda mol, **kw: np.array([1.0, 2.0]), verbose=False)
    result = wrapper(mols)
    assert result.shape == (2, 2)


def test_transform_catches_exceptions_and_returns_failed_descriptor(capsys):
    def broken_transformer(mol, **kw):
        raise ValueError("boom")

    wrapper = DescriptorWrapper(broken_transformer, verbose=False)
    result = wrapper([Chem.MolFromSmiles("CC")])
    assert isinstance(result, FailedDescriptor)
    captured = capsys.readouterr()
    assert "boom" in captured.out


def test_run_over_list_reports_progress_and_postprocesses(capsys):
    mols = [Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CCC")]
    bags = [[m] for m in mols]
    wrapper = DescriptorWrapper(lambda mol, **kw: np.array([1.0]), verbose=True)
    results, col_stats = wrapper.run(bags)
    assert len(results) == 2
    assert list(col_stats["keep_mask"]) == [True]
    captured = capsys.readouterr()
    assert "Calculating descriptors:" in captured.out


def test_run_quiet():
    bags = [[Chem.MolFromSmiles("CC")]]
    wrapper = DescriptorWrapper(lambda mol, **kw: np.array([1.0]), verbose=False)
    results, _ = wrapper.run(bags)
    assert len(results) == 1


def test_run_forwards_verbose_and_col_stats_to_postprocess(capsys):
    bags = [[Chem.MolFromSmiles("CC")]]
    wrapper = DescriptorWrapper(lambda mol, **kw: np.array([1.0, np.nan]), verbose=False)
    results, col_stats = wrapper.run(bags, verbose=True)
    assert list(col_stats["keep_mask"]) == [True, False]
    assert "Removed 1 of 2" in capsys.readouterr().out

    results_2, reused_stats = wrapper.run(bags, verbose=True, col_stats=col_stats)
    assert reused_stats is col_stats
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# postprocess
# ---------------------------------------------------------------------------

def test_postprocess_drops_any_column_with_nan(capsys):
    wrapper = DescriptorWrapper(lambda mol, **kw: None, verbose=False)
    bags = [
        np.array([[1.0, np.nan, 3.0], [1.1, 2.2, 3.3]]),
        np.array([[1.5, 2.5, 3.5], [1.6, 2.6, 3.6]]),
    ]
    cleaned, col_stats = wrapper.postprocess(bags, verbose=True)

    assert list(col_stats["keep_mask"]) == [True, False, True]
    assert cleaned[0].shape == (2, 2)
    assert cleaned[1].shape == (2, 2)

    captured = capsys.readouterr()
    assert "Removed 1 of 3" in captured.out
    assert "column 1: invalid for 1/4 conformers" in captured.out


def test_postprocess_treats_extreme_values_as_missing():
    wrapper = DescriptorWrapper(lambda mol, **kw: None, verbose=False)
    bags = [np.array([[1.0, 1e30], [1.1, 2.2]])]
    cleaned, col_stats = wrapper.postprocess(bags)
    assert list(col_stats["keep_mask"]) == [True, False]
    assert cleaned[0].shape == (2, 1)


def test_postprocess_no_removals_is_quiet_even_when_verbose(capsys):
    wrapper = DescriptorWrapper(lambda mol, **kw: None, verbose=False)
    bags = [np.array([[1.0, 2.0], [1.1, 2.1]])]
    cleaned, col_stats = wrapper.postprocess(bags, verbose=True)
    assert list(col_stats["keep_mask"]) == [True, True]
    assert capsys.readouterr().out == ""


def test_postprocess_uses_named_columns_when_available(capsys):
    class NamedTransformer:
        columns = ["alpha", "beta"]

    wrapper = DescriptorWrapper(NamedTransformer(), verbose=False)
    bags = [np.array([[1.0, np.nan]])]
    wrapper.postprocess(bags, verbose=True)
    assert "beta: invalid for 1/1 conformers" in capsys.readouterr().out


def test_postprocess_reuses_given_col_stats_without_reprinting(capsys):
    wrapper = DescriptorWrapper(lambda mol, **kw: None, verbose=False)
    train_bags = [np.array([[1.0, np.nan], [1.1, 2.2]])]
    _, col_stats = wrapper.postprocess(train_bags, verbose=True)
    capsys.readouterr()  # discard the training-time report

    val_bags = [np.array([[1.5, 9.9]])]
    cleaned, reused_stats = wrapper.postprocess(val_bags, verbose=True, col_stats=col_stats)

    assert reused_stats is col_stats
    assert cleaned[0].shape == (1, 1)
    assert capsys.readouterr().out == ""
