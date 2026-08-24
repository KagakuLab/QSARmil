import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qsarmil.conformer.base import ConformerGenerator, filter_by_energy
from qsarmil.utils.ensemble import ConformerEnsemble
from qsarmil.utils.logging import FailedConformer, FailedMolecule


def test_run_embeds_conformers_for_valid_molecules():
    gen = ConformerGenerator(num_conf=3, num_cpu=1, verbose=True)
    mol = Chem.MolFromSmiles("CCO")
    results = gen.run([mol])
    assert len(results) == 1
    assert isinstance(results[0], ConformerEnsemble)
    assert len(results[0]) == 3


def test_run_prints_progress_with_multiple_workers(capsys):
    """joblib only invokes the batch-completion print callback with more
    than one worker and enough jobs to actually batch."""
    gen = ConformerGenerator(num_conf=2, num_cpu=2, verbose=True)
    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "CCN", "CCC", "CCCl", "CCF", "CCBr"]]
    results = gen.run(mols)
    assert len(results) == 6
    captured = capsys.readouterr()
    assert "Generating conformers:" in captured.out


def test_run_applies_e_thresh_filtering():
    gen = ConformerGenerator(num_conf=10, e_thresh=1, num_cpu=1, verbose=False)
    mol = Chem.MolFromSmiles("CC(C)(C)c1ccc(cc1)CCCCCCCCCC")
    results = gen.run([mol])
    assert len(results) == 1
    assert isinstance(results[0], ConformerEnsemble)
    assert len(results[0]) <= 10


def test_run_none_input_returns_failed_conformer():
    """A None input becomes a FailedConformer(None) and is passed through
    by run() rather than raised - LazyMIL now handles such failures itself
    (see drop_failed_molecules / target_fallback in qsarmil.lazy)."""
    gen = ConformerGenerator(num_conf=3, num_cpu=1, verbose=False)
    results = gen.run([None])
    assert len(results) == 1
    assert isinstance(results[0], FailedConformer)


def test_run_mixed_batch_passes_through_failures():
    """One bad molecule in a batch doesn't stop the good ones from being
    embedded, and isn't wrapped into a ConformerEnsemble."""
    gen = ConformerGenerator(num_conf=3, num_cpu=1, verbose=False)
    mol = Chem.MolFromSmiles("CCO")
    results = gen.run([mol, None])
    assert len(results) == 2
    assert isinstance(results[0], ConformerEnsemble)
    assert isinstance(results[1], FailedConformer)


def test_generate_conformers_passes_through_failed_sentinels():
    gen = ConformerGenerator(num_conf=3, num_cpu=1, verbose=False)
    failed = FailedMolecule("garbage")
    assert gen._generate_conformers(failed) is failed
    failed_conf = FailedConformer(None)
    assert gen._generate_conformers(failed_conf) is failed_conf


def test_generate_conformers_zero_num_conf_reports_failure(capsys):
    gen = ConformerGenerator(num_conf=0, num_cpu=1, verbose=False)
    mol = Chem.MolFromSmiles("CCO")
    result = gen._generate_conformers(mol)
    assert isinstance(result, FailedConformer)


def test_seed_changes_embedding_output():
    mol_a = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    mol_b = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    gen_a = ConformerGenerator(num_conf=3, num_cpu=1, verbose=False, seed=42)
    gen_b = ConformerGenerator(num_conf=3, num_cpu=1, verbose=False, seed=123)
    ens_a = gen_a.run([mol_a])[0]
    ens_b = gen_b.run([mol_b])[0]
    coords_a = ens_a[0].GetConformer(0).GetPositions()
    coords_b = ens_b[0].GetConformer(0).GetPositions()
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
