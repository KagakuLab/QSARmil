import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qsarmil.utils.visualization import visualize_conformers_grid


def _embedded_mol(num_conf=6):
    mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    for cid in conf_ids:
        AllChem.UFFOptimizeMolecule(mol, confId=cid)
    return mol


def test_mismatched_weights_raises():
    mol = _embedded_mol(num_conf=3)
    with pytest.raises(ValueError):
        visualize_conformers_grid(mol, weights=[0.1, 0.2], key_conformers=[])


def test_show_all_true_hits_every_index_and_grey_default():
    mol = _embedded_mol(num_conf=6)
    weights = [0.5, 0.1, 0.9, 0.2, 0.3, 0.4]
    # top_n=1 -> only index 2 is PRED; key_conformers picks index 0 -> TRUE;
    # the rest (1, 3, 4, 5) fall through to the default grey branch
    visualize_conformers_grid(mol, weights, key_conformers=[0], top_n=1, show_all=True, n_cols=2)


def test_show_all_false_uses_key_and_top_union():
    mol = _embedded_mol(num_conf=6)
    weights = [0.5, 0.1, 0.9, 0.2, 0.3, 0.4]
    visualize_conformers_grid(mol, weights, key_conformers=[1], top_n=2, show_all=False)


def test_sort_by_weight_false():
    mol = _embedded_mol(num_conf=4)
    weights = [0.5, 0.1, 0.9, 0.2]
    visualize_conformers_grid(mol, weights, key_conformers=[0, 2], top_n=1, show_all=True, sort_by_weight=False)
