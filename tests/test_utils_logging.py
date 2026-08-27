
from rdkit import Chem

from qsarmil.utils.logging import FailedMolecule, OutputSuppressor


def test_failed_molecule_str_with_smiles_message():
    fm = FailedMolecule("not_a_smiles", message="SMILES parsing failed")
    assert fm.mol == "not_a_smiles"
    assert fm.message == "SMILES parsing failed"
    assert str(fm) == "not_a_smiles -> SMILES parsing failed"


def test_failed_molecule_str_with_mol():
    mol = Chem.MolFromSmiles("CCO")
    fm = FailedMolecule(mol, message="conformer generation failed")
    assert fm.mol is mol
    assert str(fm) == "CCO -> conformer generation failed"


def test_failed_molecule_default_message():
    fm = FailedMolecule("garbage")
    assert fm.message == "failed"


def test_failed_molecule_none():
    fm = FailedMolecule(None, message="conformer generation failed")
    assert fm.mol is None
    assert str(fm) == "? -> conformer generation failed"


def test_failed_molecule_str_with_bag_uses_first_conformer():
    """DescriptorWrapper stores the whole bag (list[Mol]), not a single Mol."""
    bag = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCO")]
    fm = FailedMolecule(bag, message="descriptor calculation failed")
    assert str(fm) == "CCO -> descriptor calculation failed"


def test_failed_molecule_str_with_empty_bag():
    fm = FailedMolecule([], message="descriptor calculation failed")
    assert str(fm) == "? -> descriptor calculation failed"


def test_failed_molecule_str_with_unconvertible_mol():
    """A Mol that raises during MolToSmiles still produces a readable message instead of crashing."""

    class BadMol:
        pass

    fm = FailedMolecule(BadMol(), message="conformer generation failed")
    assert str(fm) == "? -> conformer generation failed"


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
