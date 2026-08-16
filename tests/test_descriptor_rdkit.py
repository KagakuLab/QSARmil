import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from qsarmil.descriptor.rdkit import (
    RDKitAUTOCORR,
    RDKitGEOM,
    RDKitGETAWAY,
    RDKitMORSE,
    RDKitRDF,
    RDKitWHIM,
    validate_desc_vector,
)


def _embedded_mol(smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O"):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    AllChem.UFFOptimizeMolecule(mol)
    return mol


def test_validate_desc_vector_no_issues():
    x = np.array([1.0, 2.0, 3.0])
    out = validate_desc_vector(x)
    assert np.array_equal(out, x)


def test_validate_desc_vector_imputes_nan():
    x = np.array([1.0, np.nan, 3.0])
    out = validate_desc_vector(x)
    assert not np.isnan(out).any()
    assert out[1] == 2.0  # mean of 1.0 and 3.0


def test_validate_desc_vector_caps_extreme_values():
    x = np.array([1.0, 2.0, 1e30])
    out = validate_desc_vector(x)
    assert out[2] == 1.5  # mean of the two reasonable values


def test_rdkit_geom_returns_11_descriptors():
    mol = _embedded_mol()
    desc = RDKitGEOM()
    x = desc(mol, conformer_id=0)
    assert x.shape == (11,)
    assert not np.isnan(x).any()


def test_rdkit_autocorr():
    mol = _embedded_mol()
    x = RDKitAUTOCORR()(mol, conformer_id=0)
    assert x.shape == (80,)


def test_rdkit_rdf():
    mol = _embedded_mol()
    x = RDKitRDF()(mol, conformer_id=0)
    assert x.shape == (210,)


def test_rdkit_morse():
    mol = _embedded_mol()
    x = RDKitMORSE()(mol, conformer_id=0)
    assert x.shape == (224,)


def test_rdkit_whim():
    mol = _embedded_mol()
    x = RDKitWHIM()(mol, conformer_id=0)
    assert x.shape == (114,)


def test_rdkit_getaway():
    mol = _embedded_mol()
    x = RDKitGETAWAY()(mol, conformer_id=0)
    assert x.shape == (273,)
