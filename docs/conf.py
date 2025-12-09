# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the source directory to the path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "CVXPYlayers"
copyright = "2024, CVXPYlayers Authors"
author = "A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond, Z. Kolter"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "myst_parser",
    "nbsphinx",
    "sphinx_design",
    "sphinx_copybutton",
]

# MyST settings for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Autosummary settings
autosummary_generate = True

# nbsphinx settings for Jupyter notebooks
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_timeout = 300

# Intersphinx mapping for cross-references to other projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "cvxpy": ("https://www.cvxpy.org/", None),
}

# Source file settings
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "CVXPYlayers"
html_favicon = "_static/favicon.svg"
html_logo = None

# Furo theme options
html_theme_options = {
    "announcement": "<b>v1.0 Released!</b> GPU acceleration with CuClarabel + MLX support for Apple Silicon",
    "source_repository": "https://github.com/cvxpy/cvxpylayers",
    "source_branch": "master",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#2563eb",  # Blue
        "color-brand-content": "#2563eb",
        "color-admonition-background": "#f8fafc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#60a5fa",
        "color-admonition-background": "#1e293b",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/cvxpy/cvxpylayers",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/cvxpylayers/",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm3.5 7.5a.5.5 0 0 1 0 1H5.707l2.147 2.146a.5.5 0 0 1-.708.708l-3-3a.5.5 0 0 1 0-.708l3-3a.5.5 0 1 1 .708.708L5.707 7.5H11.5z"/>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Pygments syntax highlighting
pygments_style = "default"
pygments_dark_style = "monokai"

# Custom CSS
html_css_files = ["custom.css"]

# -- Options for autodoc -----------------------------------------------------
# Only mock MLX (Apple Silicon only, not available on Intel)
autodoc_mock_imports = ["mlx"]

# Suppress specific warnings
suppress_warnings = ["autodoc.import_object"]
