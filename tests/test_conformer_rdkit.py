from rdkit import Chem

from qsarmil.conformer.rdkit import RDKitConformerGenerator
from qsarmil.utils.ensemble import ConformerEnsemble


def test_rdkit_conformer_generator_forwards_params():
    gen = RDKitConformerGenerator(num_conf=3, e_thresh=2, num_cpu=1, verbose=False, seed=7)
    assert gen.num_conf == 3
    assert gen.e_thresh == 2
    assert gen.num_cpu == 1
    assert gen.verbose is False
    assert gen.seed == 7


def test_rdkit_conformer_generator_runs():
    gen = RDKitConformerGenerator(num_conf=3, num_cpu=1, verbose=False)
    mol = Chem.MolFromSmiles("CCO")
    results = gen.run([mol])
    assert len(results) == 1
    assert isinstance(results[0], ConformerEnsemble)
    assert len(results[0]) == 3
