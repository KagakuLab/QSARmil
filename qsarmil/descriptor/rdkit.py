from __future__ import annotations

import numpy as np
from rdkit.Chem import Descriptors3D, Mol


def validate_desc_vector(x: np.ndarray) -> np.ndarray:
    """Validate and clean a descriptor vector.

    Replaces NaN values with the mean of valid elements and caps extreme values
    beyond 1e25 by replacing them with the mean of reasonable elements.

    Args:
        x (np.ndarray): Descriptor vector to validate.

    Returns:
        np.ndarray: Cleaned descriptor vector.
    """
    # nan values
    if np.isnan(x).sum() > 0:
        imp = np.mean(x[~np.isnan(x)])
        x = np.where(np.isnan(x), imp, x)  # TODO temporary solution, should be revised
    # extreme dsc values
    if (abs(x) >= 10**25).sum() > 0:
        imp = np.mean(x[abs(x) <= 10**25])
        x = np.where(abs(x) <= 10**25, x, imp)
    return x


class RDKitDescriptor3D:
    """Base class to compute 3D molecular descriptors using RDKit.

    Args:
        desc_name (str, optional): Name of the 3D descriptor function from RDKit Descriptors3D.
    """

    def __init__(self, desc_name: str | None = None) -> None:
        """Look up the RDKit descriptor function to use.

        Args:
            desc_name (str, optional): Name of a function on
                ``rdkit.Chem.Descriptors3D.rdMolDescriptors``. Left unset by
                subclasses that override ``__call__`` themselves (e.g.
                :class:`RDKitGEOM`).
        """
        super().__init__()

        if desc_name:
            self.transformer = getattr(Descriptors3D.rdMolDescriptors, desc_name)

    def __call__(self, mol: Mol, conformer_id: int | None = None) -> np.ndarray:
        """Compute the 3D descriptor for a molecule and optional conformer.

        Args:
            mol (rdkit.Chem.Mol): Molecule to compute descriptors for.
            conformer_id (int, optional): Specific conformer ID to use.

        Returns:
            np.ndarray: Validated descriptor vector.
        """
        x = np.array(self.transformer(mol, confId=conformer_id))
        x = validate_desc_vector(x)
        return x


class RDKitGEOM(RDKitDescriptor3D):
    """Compute multiple 3D geometric descriptors for a molecule.

    Computes descriptors such as asphericity, eccentricity, PMI, radius
    of gyration, etc.
    """

    def __init__(self) -> None:
        """Initialize the RDKitGEOM descriptor with a fixed list of geometric
        descriptors."""
        super().__init__()

        self.columns = [
            "CalcAsphericity",
            "CalcEccentricity",
            "CalcInertialShapeFactor",
            "CalcNPR1",
            "CalcNPR2",
            "CalcPMI1",
            "CalcPMI2",
            "CalcPMI3",
            "CalcRadiusOfGyration",
            "CalcSpherocityIndex",
            "CalcPBF",
        ]

    def __call__(self, mol: Mol, conformer_id: int | None = None) -> np.ndarray:
        """Compute all geometric descriptors for a molecule and optional
        conformer.

        Args:
            mol (rdkit.Chem.Mol): Molecule to compute descriptors for.
            conformer_id (int, optional): Specific conformer ID to use.

        Returns:
            np.ndarray: Validated geometric descriptor vector.
        """
        x = []
        for desc_name in self.columns:
            transformer = getattr(Descriptors3D.rdMolDescriptors, desc_name)
            x.append(transformer(mol, confId=conformer_id))
        x = np.array(x)
        x = validate_desc_vector(x)
        return x


class RDKitAUTOCORR(RDKitDescriptor3D):
    """Compute 3D autocorrelation descriptors for a molecule."""

    def __init__(self) -> None:
        """Wire up RDKit's ``CalcAUTOCORR3D``."""
        super().__init__("CalcAUTOCORR3D")


class RDKitRDF(RDKitDescriptor3D):
    """Compute 3D radial distribution function (RDF) descriptors for a
    molecule."""

    def __init__(self) -> None:
        """Wire up RDKit's ``CalcRDF``."""
        super().__init__("CalcRDF")


class RDKitMORSE(RDKitDescriptor3D):
    """Compute 3D Morse descriptors for a molecule."""

    def __init__(self) -> None:
        """Wire up RDKit's ``CalcMORSE``."""
        super().__init__("CalcMORSE")


class RDKitWHIM(RDKitDescriptor3D):
    """Compute 3D WHIM descriptors for a molecule."""

    def __init__(self) -> None:
        """Wire up RDKit's ``CalcWHIM``."""
        super().__init__("CalcWHIM")


class RDKitGETAWAY(RDKitDescriptor3D):
    """Compute 3D GETAWAY descriptors for a molecule."""

    def __init__(self) -> None:
        """Wire up RDKit's ``CalcGETAWAY``."""
        super().__init__("CalcGETAWAY")
