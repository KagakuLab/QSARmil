from __future__ import annotations

from qsarmil.conformer.base import ConformerGenerator


class RDKitConformerGenerator(ConformerGenerator):
    """Generate RDKit 3D conformers for molecules using the ETKDG method.

    Inherits from ConformerGenerator and implements RDKit-specific molecule
    preparation and conformer embedding.

    Args:
        num_conf (int): Number of conformers to generate per molecule.
        e_thresh (float, optional): Energy threshold for filtering high-energy conformers.
        num_cpu (int): Number of CPU threads to use for parallel processing.
        verbose (bool): Whether to display a progress bar during generation.
        seed (int): Random seed for conformer embedding.
    """

    def __init__(
        self,
        num_conf: int = 10,
        e_thresh: float | None = None,
        num_cpu: int = 1,
        verbose: bool = True,
        seed: int = 42,
    ) -> None:
        """Initialize RDKitConformerGenerator with generation parameters.

        Args:
            num_conf (int): Number of conformers to generate per molecule.
            e_thresh (float, optional): Energy threshold for filtering high-energy conformers.
            num_cpu (int): Number of CPU threads to use for parallel processing.
            verbose (bool): Whether to display a progress bar during generation.
            seed (int): Random seed for conformer embedding.
        """
        super().__init__(num_conf=num_conf, e_thresh=e_thresh, num_cpu=num_cpu, verbose=verbose, seed=seed)
