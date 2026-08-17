from rdkit import Chem

from qsarmil.data.input_data import DataValidator


def test_validate_one_success():
    validator = DataValidator(num_cpu=1, verbose=False)
    result = validator._validate_one("CCO")
    assert result["is_valid_smiles"] is True
    assert result["sanitization_passed"] is True
    assert result["conformer_generated"] is True
    assert result["error"] is None


def test_validate_one_parsing_failure():
    validator = DataValidator(num_cpu=1, verbose=False)
    result = validator._validate_one("not_a_valid_smiles!!!")
    assert result["is_valid_smiles"] is False
    assert result["error"] == "SMILES parsing failed"


def test_validate_one_sanitization_failure(monkeypatch):
    validator = DataValidator(num_cpu=1, verbose=False)

    def raise_sanitize(mol):
        raise ValueError("boom")

    monkeypatch.setattr(Chem, "SanitizeMol", raise_sanitize)
    result = validator._validate_one("CCO")
    assert result["is_valid_smiles"] is True
    assert result["sanitization_passed"] is False
    assert "Sanitization failed" in result["error"]


def test_validate_one_addhs_failure(monkeypatch):
    validator = DataValidator(num_cpu=1, verbose=False)

    def raise_addhs(mol):
        raise ValueError("boom")

    monkeypatch.setattr(Chem, "AddHs", raise_addhs)
    result = validator._validate_one("CCO")
    assert result["sanitization_passed"] is True
    assert "AddHs failed" in result["error"]


def test_validate_one_embedding_failure(monkeypatch):
    from rdkit.Chem import AllChem

    validator = DataValidator(num_cpu=1, verbose=False)
    monkeypatch.setattr(AllChem, "EmbedMolecule", lambda mol, params: -1)
    result = validator._validate_one("CCO")
    assert result["conformer_generated"] is False
    assert result["error"] == "Conformer embedding failed"


def test_validate_one_embedding_exception(monkeypatch):
    from rdkit.Chem import AllChem

    validator = DataValidator(num_cpu=1, verbose=False)

    def raise_embed(mol, params):
        raise ValueError("boom")

    monkeypatch.setattr(AllChem, "EmbedMolecule", raise_embed)
    result = validator._validate_one("CCO")
    assert "Embedding exception" in result["error"]


def test_validate_smiles_parallel():
    validator = DataValidator(num_cpu=1, verbose=False)
    results = validator.validate_smiles(["CCO", "not_a_valid_smiles!!!"])
    assert len(results) == 2
    assert results[0]["conformer_generated"] is True
    assert results[1]["conformer_generated"] is False


def test_filter_dataframe_removes_invalid_rows(capsys):
    import pandas as pd

    validator = DataValidator(num_cpu=1, verbose=True)
    df = pd.DataFrame({0: ["CCO", "not_a_valid_smiles!!!", "c1ccccc1"], 1: [1.0, 2.0, 3.0]})
    filtered = validator.filter_dataframe(df)
    assert len(filtered) == 2
    assert list(filtered[0]) == ["CCO", "c1ccccc1"]
    captured = capsys.readouterr()
    assert "Removed rows" in captured.out


def test_filter_dataframe_no_removals(capsys):
    import pandas as pd

    validator = DataValidator(num_cpu=1, verbose=True)
    df = pd.DataFrame({0: ["CCO", "c1ccccc1"], 1: [1.0, 2.0]})
    filtered = validator.filter_dataframe(df)
    assert len(filtered) == 2
    captured = capsys.readouterr()
    assert "No rows removed" in captured.out


def test_filter_dataframe_quiet():
    import pandas as pd

    validator = DataValidator(num_cpu=1, verbose=False)
    df = pd.DataFrame({0: ["not_a_valid_smiles!!!"], 1: [1.0]})
    filtered = validator.filter_dataframe(df)
    assert len(filtered) == 0


def test_seed_affects_trial_embedding():
    validator_a = DataValidator(num_cpu=1, verbose=False, seed=42)
    validator_b = DataValidator(num_cpu=1, verbose=False, seed=123)
    assert validator_a.seed == 42
    assert validator_b.seed == 123
