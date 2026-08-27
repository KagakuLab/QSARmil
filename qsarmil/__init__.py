from importlib.metadata import PackageNotFoundError, version

from qsarmil.modelling.meta import MultiConformerClassifier, MultiConformerEstimator, MultiConformerRegressor

try:
    __version__ = version("qsarmil")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["MultiConformerClassifier", "MultiConformerEstimator", "MultiConformerRegressor", "__version__"]
