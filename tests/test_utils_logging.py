
from rdkit import Chem

from qsarmil.utils.logging import (
    FailedConformer,
    FailedDescriptor,
    FailedMolecule,
    OutputSuppressor,
)


def test_failed_molecule_str():
    fm = FailedMolecule("not_a_smiles")
    assert fm.smiles == "not_a_smiles"
    assert str(fm) == "not_a_smiles -> SMILES parsing failed"


def test_failed_conformer_str():
    mol = Chem.MolFromSmiles("CCO")
    fc = FailedConformer(mol)
    assert fc.mol is mol
    assert str(fc) == "CCO -> conformer generation failed"


def test_failed_conformer_none():
    fc = FailedConformer(None)
    assert fc.mol is None


def test_failed_descriptor_str():
    mol = Chem.MolFromSmiles("CCO")
    fd = FailedDescriptor(mol)
    assert str(fd) == "CCO -> descriptor calculation failed"


def test_output_suppressor_suppresses_and_restores(capsys):
    print("before")
    with OutputSuppressor():
        print("suppressed")
    print("after")

    captured = capsys.readouterr()
    assert "before" in captured.out
    assert "after" in captured.out
    assert "suppressed" not in captured.out


def test_output_suppressor_nested(capsys):
    """Nested use should only restore output once the outermost exits."""
    with OutputSuppressor():
        with OutputSuppressor():
            print("inner suppressed")
        print("still suppressed (outer still active)")
    print("visible again")

    captured = capsys.readouterr()
    assert "inner suppressed" not in captured.out
    assert "still suppressed" not in captured.out
    assert "visible again" in captured.out
