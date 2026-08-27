import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from qsarmil.descriptor.rdkit import RDKitGEOM
from qsarmil.descriptor.wrapper import DescriptorWrapper
from qsarmil.utils.logging import FailedMolecule


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


def test_transform_catches_exceptions_and_returns_failed_molecule(capsys):
    def broken_transformer(mol, **kw):
        raise ValueError("boom")

    wrapper = DescriptorWrapper(broken_transformer, verbose=False)
    result = wrapper([Chem.MolFromSmiles("CC")])
    assert isinstance(result, FailedMolecule)
    assert result.message == "descriptor calculation failed"
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


def test_run_passes_through_failed_bag_instead_of_raising(capsys):
    """One bad molecule's bag becomes a FailedMolecule sentinel; the rest of the batch still succeeds."""

    def broken_transformer(mol, **kw):
        if mol.GetNumAtoms() == 1:  # fail only for a specific bag
            raise ValueError("degenerate geometry")
        return np.array([1.0, 2.0])

    wrapper = DescriptorWrapper(broken_transformer, verbose=False)
    bags = [[Chem.MolFromSmiles("CC")], [Chem.MolFromSmiles("C")]]
    results = wrapper.run(bags)

    assert isinstance(results[0], np.ndarray)
    assert isinstance(results[1], FailedMolecule)
    assert "degenerate geometry" in capsys.readouterr().out


def test_run_all_bags_failed_returns_them_unchanged():
    def broken_transformer(mol, **kw):
        raise ValueError("boom")

    wrapper = DescriptorWrapper(broken_transformer, verbose=False)
    results = wrapper.run([[Chem.MolFromSmiles("CC")]])
    assert isinstance(results[0], FailedMolecule)


# ---------------------------------------------------------------------------
# NaN/extreme-value column dropping (computed fresh on every call, not persisted)
# ---------------------------------------------------------------------------

def test_run_drops_column_with_nan_and_reports_it(capsys):
    outputs = iter([np.array([1.0, np.nan]), np.array([1.1, 2.2]), np.array([1.5, 2.5])])
    wrapper = DescriptorWrapper(lambda mol, **kw: next(outputs), verbose=False)
    bags = [[Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CC")], [Chem.MolFromSmiles("CCC")]]

    results = wrapper.run(bags)

    assert results[0].shape == (2, 1)
    assert results[1].shape == (1, 1)
    assert "Removed 1 of 2 descriptor column(s)" in capsys.readouterr().out


def test_run_treats_extreme_values_as_missing():
    outputs = iter([np.array([1.0, 1e30]), np.array([1.1, 2.2])])
    wrapper = DescriptorWrapper(lambda mol, **kw: next(outputs), verbose=False)
    bags = [[Chem.MolFromSmiles("CC")], [Chem.MolFromSmiles("CCC")]]

    results = wrapper.run(bags)

    assert results[0].shape == (1, 1)
    assert results[1].shape == (1, 1)


def test_run_no_removals_prints_nothing(capsys):
    wrapper = DescriptorWrapper(lambda mol, **kw: np.array([1.0, 2.0]), verbose=False)
    bags = [[Chem.MolFromSmiles("CC")], [Chem.MolFromSmiles("CCC")]]

    results = wrapper.run(bags)

    assert results[0].shape == (1, 2)
    assert capsys.readouterr().out == ""


def test_run_recomputes_keep_mask_every_call_instead_of_persisting():
    """Unlike the old design, nothing is remembered between calls - each run() call computes its
    own keep/drop decision from scratch, since the whole dataset now always goes through in one call."""

    outputs_first = iter([np.array([1.0, np.nan]), np.array([1.1, 2.2])])
    wrapper = DescriptorWrapper(lambda mol, **kw: next(outputs_first), verbose=False)
    results1 = wrapper.run([[Chem.MolFromSmiles("CC")], [Chem.MolFromSmiles("CCC")]])
    assert results1[0].shape == (1, 1)  # one column dropped, based on this call's data

    # a later call with clean data of its own keeps both columns - the earlier drop isn't remembered
    outputs_second = iter([np.array([9.0, 9.0])])
    wrapper.transformer = lambda mol, **kw: next(outputs_second)
    results2 = wrapper.run([[Chem.MolFromSmiles("CCCC")]])

    assert results2[0].shape == (1, 2)
