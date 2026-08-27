from rdkit import Chem
from rdkit.Chem import BRICS

from qsarmil.fragment.rdkit import RDKitFragmentGenerator
from qsarmil.utils.logging import FailedMolecule


def test_generate_fragments_success():
    gen = RDKitFragmentGenerator(verbose=False)
    mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(cc1)OCC")  # phenacetin, BRICS-decomposable
    result = gen._transform(mol)
    assert isinstance(result, list)
    assert len(result) >= 1


def test_generate_fragments_passes_through_failed_sentinel(capsys):
    gen = RDKitFragmentGenerator(verbose=False)
    failed = FailedMolecule("garbage", message="SMILES parsing failed")
    assert gen._transform(failed) is failed
    captured = capsys.readouterr()
    assert "Failed molecule" in captured.out


def test_generate_fragments_falls_back_on_exception(monkeypatch, capsys):
    gen = RDKitFragmentGenerator(verbose=False)
    mol = Chem.MolFromSmiles("CCO")

    def raise_decompose(mol):
        raise ValueError("boom")

    monkeypatch.setattr(BRICS, "BRICSDecompose", raise_decompose)
    result = gen._transform(mol)
    assert isinstance(result, list)
    assert list(result) == [mol]
    captured = capsys.readouterr()
    assert "boom" in captured.out


def test_run_reports_progress(capsys):
    gen = RDKitFragmentGenerator(verbose=True)
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC(=O)Nc1ccc(cc1)OCC")]
    results = gen.run(mols)
    assert len(results) == 2
    captured = capsys.readouterr()
    assert "Generating fragments:" in captured.out


def test_run_quiet():
    gen = RDKitFragmentGenerator(verbose=False)
    results = gen.run([Chem.MolFromSmiles("CCO")])
    assert len(results) == 1
