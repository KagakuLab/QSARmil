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


def test_run_returns_plain_list_of_bags(capsys):
    mols = [Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CCC")]
    bags = [[m] for m in mols]
    wrapper = DescriptorWrapper(lambda mol, **kw: np.array([1.0]), verbose=True)
    results = wrapper.run(bags)
    assert isinstance(results, list)
    assert len(results) == 2
    captured = capsys.readouterr()
    assert "Calculating descriptors:" in captured.out


def test_run_quiet():
    bags = [[Chem.MolFromSmiles("CC")]]
    wrapper = DescriptorWrapper(lambda mol, **kw: np.array([1.0]), verbose=False)
    results = wrapper.run(bags)
    assert len(results) == 1


def test_run_raises_clear_error_for_failed_descriptor():
    def broken_transformer(mol, **kw):
        raise ValueError("boom")

    wrapper = DescriptorWrapper(broken_transformer, verbose=False)
    with pytest.raises(ValueError, match=r"failed for 1 of 1 molecule\(s\)"):
        wrapper.run([[Chem.MolFromSmiles("CC")]])


def test_run_error_message_includes_smiles_and_row():
    def broken_transformer(mol, **kw):
        raise ValueError("boom")

    wrapper = DescriptorWrapper(broken_transformer, verbose=False)
    with pytest.raises(ValueError, match=r"Row 0: CCO -> descriptor calculation failed"):
        wrapper.run([[Chem.MolFromSmiles("CCO")]])


def test_run_raises_clear_error_instead_of_crashing_downstream():
    """One bad molecule surfaces as this clear error immediately, not as a cryptic
    shape-mismatch error later on when the caller tries to stack/scale the bags."""

    def broken_transformer(mol, **kw):
        if mol.GetNumAtoms() == 1:  # fail only for a specific bag
            raise ValueError("degenerate geometry")
        return np.array([1.0, 2.0])

    wrapper = DescriptorWrapper(broken_transformer, verbose=False)
    bags = [[Chem.MolFromSmiles("CC")], [Chem.MolFromSmiles("C")]]
    with pytest.raises(ValueError, match=r"failed for 1 of 2 molecule\(s\)"):
        wrapper.run(bags)


# ---------------------------------------------------------------------------
# Internal NaN/extreme-value column dropping (learned once, reused silently)
# ---------------------------------------------------------------------------

def test_run_drops_column_with_nan_and_reports_it(capsys):
    outputs = iter([np.array([1.0, np.nan]), np.array([1.1, 2.2]), np.array([1.5, 2.5])])
    wrapper = DescriptorWrapper(lambda mol, **kw: next(outputs), verbose=False)
    bags = [[Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CC")], [Chem.MolFromSmiles("CCC")]]

    results = wrapper.run(bags)

    assert results[0].shape == (2, 1)
    assert results[1].shape == (1, 1)
    captured = capsys.readouterr()
    assert "Removed 1 of 2" in captured.out
    assert "column 1: invalid for 1/3 conformers" in captured.out


def test_run_treats_extreme_values_as_missing():
    outputs = iter([np.array([1.0, 1e30]), np.array([1.1, 2.2])])
    wrapper = DescriptorWrapper(lambda mol, **kw: next(outputs), verbose=False)
    bags = [[Chem.MolFromSmiles("CC")], [Chem.MolFromSmiles("CCC")]]

    results = wrapper.run(bags)

    assert results[0].shape == (1, 1)
    assert results[1].shape == (1, 1)


def test_run_no_removals_prints_nothing():
    wrapper = DescriptorWrapper(lambda mol, **kw: np.array([1.0, 2.0]), verbose=False)
    bags = [[Chem.MolFromSmiles("CC")], [Chem.MolFromSmiles("CCC")]]

    results = wrapper.run(bags)

    assert results[0].shape == (1, 2)


def test_run_reuses_learned_keep_mask_on_later_calls_without_reprinting(capsys):
    """The keep/drop decision is made once (on the first call) and silently reused after -
    this is what lets train/val/test end up with the same columns without col_stats."""

    outputs_first = iter([np.array([1.0, np.nan]), np.array([1.1, 2.2])])
    wrapper = DescriptorWrapper(lambda mol, **kw: next(outputs_first), verbose=False)
    results1 = wrapper.run([[Chem.MolFromSmiles("CC")], [Chem.MolFromSmiles("CCC")]])
    assert results1[0].shape == (1, 1)
    capsys.readouterr()  # discard the first call's report

    # second call: even if this batch alone would look "clean", the earlier decision still applies
    outputs_second = iter([np.array([9.0, 9.0])])
    wrapper.transformer = lambda mol, **kw: next(outputs_second)
    results2 = wrapper.run([[Chem.MolFromSmiles("CCCC")]])

    assert results2[0].shape == (1, 1)
    assert capsys.readouterr().out == ""


def test_report_removed_columns_uses_named_columns_when_available(capsys):
    class NamedTransformer:
        columns = ["alpha", "beta"]

        def __call__(self, mol, **kw):
            return np.array([1.0, np.nan])

    wrapper = DescriptorWrapper(NamedTransformer(), verbose=False)
    wrapper.run([[Chem.MolFromSmiles("CC")]])
    assert "beta: invalid for 1/1 conformers" in capsys.readouterr().out
