import numpy as np

from qsarmil.utils.ensemble import ConformerEnsemble, FragmentEnsemble, MixtureEnsemble
from qsarmil.utils.logging import FailedDescriptor


class DescriptorWrapper:
    """Wrapper to compute molecular descriptors for multiple conformers in
    parallel.

    Converts a molecule into a "bag" of descriptor vectors, one per conformer,
    with optional parallelization and progress tracking.

    Args:
        transformer (callable): Descriptor function or object that accepts a molecule
            and optional conformer ID, returning a descriptor vector.
        verbose (bool): Whether to display a progress bar.
    """

    def __init__(self, transformer, verbose=True):
        """Initialize the descriptor wrapper.

        Args:
            transformer (callable): Descriptor function or object.
            verbose (bool): Whether to show progress bar.
        """
        super().__init__()
        self.transformer = transformer
        self.verbose = verbose

    def __call__(self, mol, *args, **kwargs):
        """Compute the descriptor bag for a single molecule.

        Args:
            mol: A conformer/fragment/mixture ensemble to compute descriptors for.

        Returns:
            np.ndarray: One descriptor vector per instance in the ensemble.
        """
        return self._transform(mol)

    def _ensemble_to_descriptors(self, ensemble_of_instances):
        """Convert a molecule into a bag of descriptor vectors."""

        bag = []
        if isinstance(ensemble_of_instances, ConformerEnsemble):
            for conf in ensemble_of_instances:
                x = self.transformer(conf, conformer_id=0)
                bag.append(x.flatten())

        elif isinstance(ensemble_of_instances, FragmentEnsemble):
            for frag in ensemble_of_instances:
                x = self.transformer(frag)
                bag.append(x.flatten())

        elif isinstance(ensemble_of_instances, MixtureEnsemble):
            for comp in ensemble_of_instances:
                x = self.transformer(comp)
                bag.append(x.flatten())

        else:
            raise TypeError(f"Unsupported type {type(ensemble_of_instances)}")

        return np.array(bag)

    def _transform(self, mol):
        """Compute descriptors for a single molecule."""
        try:
            x = self._ensemble_to_descriptors(mol)
        except Exception as e:
            print(e)
            x = FailedDescriptor(mol)
        return x

    def run(self, list_of_mols):
        """Compute descriptors for a list of molecules."""

        total = len(list_of_mols)
        results = []
        for i, mol in enumerate(list_of_mols, 1):
            results.append(self._transform(mol))
            if self.verbose:
                print(f"Calculating descriptors: {i}/{total}", end="\r", flush=True)

        if self.verbose:
            print(f"Calculating descriptors: {total}/{total}")

        return results
