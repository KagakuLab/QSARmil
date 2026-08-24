from __future__ import annotations

import numpy as np
from rdkit.Chem import Descriptors3D, Mol


class RDKitDescriptor3D:
    """Base class to compute 3D molecular descriptors using RDKit."""

    def __init__(self, desc_name: str | None = None) -> None:
        """Look up the RDKit descriptor function to use, if given (subclasses like RDKitGEOM set it themselves)."""
        super().__init__()

        if desc_name:
            self.transformer = getattr(Descriptors3D.rdMolDescriptors, desc_name)

    def __call__(self, mol: Mol, conformer_id: int | None = None) -> np.ndarray:
        """Compute the raw, uncleaned 3D descriptor for a molecule and optional conformer.

        Args:
            mol (rdkit.Chem.Mol): Molecule to compute descriptors for.
            conformer_id (int, optional): Specific conformer ID to use.

        Returns:
            np.ndarray: Raw, uncleaned descriptor vector.
        """
        return np.array(self.transformer(mol, confId=conformer_id))


class RDKitGEOM(RDKitDescriptor3D):
    """Compute 3D geometric descriptors (asphericity, eccentricity, PMI, radius of gyration, etc.) for a molecule."""

    def __init__(self) -> None:
        """Set the fixed list of geometric descriptor names to compute."""
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
        """Compute all geometric descriptors for a molecule and optional conformer.

        Args:
            mol (rdkit.Chem.Mol): Molecule to compute descriptors for.
            conformer_id (int, optional): Specific conformer ID to use.

        Returns:
            np.ndarray: Raw, uncleaned geometric descriptor vector.
        """
        x = []
        for desc_name in self.columns:
            transformer = getattr(Descriptors3D.rdMolDescriptors, desc_name)
            x.append(transformer(mol, confId=conformer_id))
        return np.array(x)


class RDKitAUTOCORR(RDKitDescriptor3D):
    """Compute 3D autocorrelation descriptors for a molecule."""

    def __init__(self) -> None:
        """Wire up RDKit's ``CalcAUTOCORR3D``."""
        super().__init__("CalcAUTOCORR3D")


class RDKitRDF(RDKitDescriptor3D):
    """Compute 3D radial distribution function (RDF) descriptors for a molecule."""

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
