# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "QSARmil"
copyright = "2026, KagakuLab"
author = "Dmitry"
# Keep this in sync with qsarmil/__init__.py or pyproject.toml version.
release = "1.0.0"

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",  # allows mixing Markdown (.md) with reStructuredText (.rst)
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Recognize both .rst and .md as source files.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output --------------------------------------------------

html_theme = "alabaster"
html_static_path = ["_static"]

# Alabaster-specific sidebar / theme options.
# Full list: https://alabaster.readthedocs.io/en/latest/customization.html
html_theme_options = {
    "description": "Molecular multi-instance machine learning for QSAR modeling",
    "github_user": "KagakuLab",
    "github_repo": "qsarmil",
    "github_button": True,
    "github_type": "star",
    "fixed_sidebar": True,
    "sidebar_collapse": True,
    "page_width": "1100px",
}

html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
    ]
}
