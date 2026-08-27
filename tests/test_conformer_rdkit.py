import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qsarmil.conformer.rdkit import RDKitConformerGenerator, filter_by_energy
from qsarmil.utils.logging import FailedMolecule


def test_rdkit_conformer_generator_forwards_params():
    gen = RDKitConformerGenerator(num_conf=3, e_thresh=2, num_cpu=1, verbose=False, random_seed=7)
    assert gen.num_conf == 3
    assert gen.e_thresh == 2
    assert gen.num_cpu == 1
    assert gen.verbose is False
    assert gen.random_seed == 7


def test_split_into_conformers_one_mol_per_embedded_conformer():
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMultipleConfs(mol, numConfs=3, params=params)

    gen = RDKitConformerGenerator(num_conf=3, num_cpu=1, verbose=False)
    bag = gen._split_into_conformers(mol)

    assert isinstance(bag, list)
    assert len(bag) == 3
    for conf_mol in bag:
        assert conf_mol.GetNumConformers() == 1


def test_run_embeds_conformers_for_valid_molecules():
    gen = RDKitConformerGenerator(num_conf=3, num_cpu=1, verbose=True)
    mol = Chem.MolFromSmiles("CCO")
    results = gen.run([mol])
    assert len(results) == 1
    assert isinstance(results[0], list)
    assert len(results[0]) == 3


def test_run_prints_progress_with_multiple_workers(capsys):
    """joblib only invokes the batch-completion print callback with more than one worker and enough jobs."""
    gen = RDKitConformerGenerator(num_conf=2, num_cpu=2, verbose=True)
    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "CCN", "CCC", "CCCl", "CCF", "CCBr"]]
    results = gen.run(mols)
    assert len(results) == 6
    captured = capsys.readouterr()
    assert "Generating conformers:" in captured.out


def test_run_applies_e_thresh_filtering():
    gen = RDKitConformerGenerator(num_conf=10, e_thresh=1, num_cpu=1, verbose=False)
    mol = Chem.MolFromSmiles("CC(C)(C)c1ccc(cc1)CCCCCCCCCC")
    results = gen.run([mol])
    assert len(results) == 1
    assert isinstance(results[0], list)
    assert len(results[0]) <= 10


def test_run_none_input_becomes_failed_molecule():
    """A bare None (e.g. from a failed Chem.MolFromSmiles upstream) is passed through, not raised."""
    gen = RDKitConformerGenerator(num_conf=3, num_cpu=1, verbose=False)
    results = gen.run([None])
    assert len(results) == 1
    assert isinstance(results[0], FailedMolecule)


def test_run_mixed_batch_passes_through_failures():
    """One bad molecule in a batch doesn't stop the good ones from being embedded."""
    gen = RDKitConformerGenerator(num_conf=3, num_cpu=1, verbose=False)
    mol = Chem.MolFromSmiles("CCO")
    results = gen.run([mol, None])
    assert len(results) == 2
    assert isinstance(results[0], list)
    assert isinstance(results[1], FailedMolecule)


def test_generate_conformers_passes_through_failed_molecule():
    gen = RDKitConformerGenerator(num_conf=3, num_cpu=1, verbose=False)
    failed = FailedMolecule("garbage", message="SMILES parsing failed")
    assert gen._generate_conformers(failed) is failed


def test_generate_conformers_none_becomes_failed_molecule():
    gen = RDKitConformerGenerator(num_conf=3, num_cpu=1, verbose=False)
    result = gen._generate_conformers(None)
    assert isinstance(result, FailedMolecule)
    assert result.message == "SMILES parsing failed"


def test_generate_conformers_zero_num_conf_reports_failure():
    gen = RDKitConformerGenerator(num_conf=0, num_cpu=1, verbose=False)
    mol = Chem.MolFromSmiles("CCO")
    result = gen._generate_conformers(mol)
    assert isinstance(result, FailedMolecule)
    assert result.message == "conformer generation failed"


def test_random_seed_changes_embedding_output():
    mol_a = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    mol_b = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    gen_a = RDKitConformerGenerator(num_conf=3, num_cpu=1, verbose=False, random_seed=42)
    gen_b = RDKitConformerGenerator(num_conf=3, num_cpu=1, verbose=False, random_seed=123)
    bag_a = gen_a.run([mol_a])[0]
    bag_b = gen_b.run([mol_b])[0]
    coords_a = bag_a[0].GetConformer(0).GetPositions()
    coords_b = bag_b[0].GetConformer(0).GetPositions()
    assert (coords_a != coords_b).any()


def _embed(smiles, num_conf=5):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    for cid in conf_ids:
        AllChem.UFFOptimizeMolecule(mol, confId=cid)
    return mol


def test_filter_by_energy_keeps_low_energy_conformers():
    mol = _embed("CC(C)(C)c1ccc(cc1)CCCCCCCCCC", num_conf=8)
    before = mol.GetNumConformers()
    filtered = filter_by_energy(mol, e_thresh=1)
    assert filtered.GetNumConformers() <= before
    assert filtered.GetNumConformers() >= 1


def test_filter_by_energy_raises_when_uff_fails_for_every_conformer(monkeypatch):
    mol = _embed("CCO", num_conf=3)
    monkeypatch.setattr(AllChem, "UFFGetMoleculeForceField", lambda *a, **kw: None)
    with pytest.raises(IndexError):
        filter_by_energy(mol, e_thresh=1)
