import qsarmil


def test_top_level_reexports():
    assert qsarmil.MultiConformerEstimator is not None
    assert qsarmil.MultiConformerRegressor is not None
    assert qsarmil.MultiConformerClassifier is not None
    assert issubclass(qsarmil.MultiConformerRegressor, qsarmil.MultiConformerEstimator)
    assert issubclass(qsarmil.MultiConformerClassifier, qsarmil.MultiConformerEstimator)


def test_version_is_a_non_empty_string():
    assert isinstance(qsarmil.__version__, str)
    assert qsarmil.__version__ != ""


def test_version_falls_back_when_package_metadata_missing(monkeypatch):
    import importlib

    def raise_not_found(name):
        from importlib.metadata import PackageNotFoundError

        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", raise_not_found)
    importlib.reload(qsarmil)
    try:
        assert qsarmil.__version__ == "0.0.0+unknown"
    finally:
        importlib.reload(qsarmil)  # restore normal state for other tests
