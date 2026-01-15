# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# Add the project root to the path so autodoc can find the package
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'shepherd-score'
copyright = f'2024-{datetime.now().year}, Kento Abeywardane'
author = 'Kento Abeywardane'

# The version info
from shepherd_score import __version__  # noqa: E402
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'myst_nb',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The master toctree document
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
pygments_style = 'sphinx'

# Set HTML title without version number
html_title = 'shepherd-score'
html_logo = '_static/logo.svg'

html_theme_options = {
    'repository_url': 'https://github.com/coleygroup/shepherd-score',
    'path_to_docs': 'docs',
    'use_source_button': True,
    'use_download_button': True,
    'use_repository_button': True,
    'use_issues_button': True,
    'logo': {
        'image_light': '_static/logo.svg',
        'image_dark': '_static/logo.svg',
        'text': 'shepherd-score',
    },
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/coleygroup/shepherd-score',
            'icon': 'fa-brands fa-square-github',
            'type': 'fontawesome',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/shepherd-score/',
            'icon': 'fa-solid fa-box',
            'type': 'fontawesome',
        },
    ],
}

html_static_path = ['_static']
html_css_files = ['custom.css']

# copybutton settings
copybutton_exclude = '.linenos, .gp'

# -- Extension configuration -------------------------------------------------

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}
autodoc_typehints = 'description'
autodoc_mock_imports = [
    'torch',
    'jax',
    'jaxlib',
    'optax',
    'open3d',
    'rdkit',
    'meeko',
    'vina',
    'openbabel',
    'prolif',
    'biopython',
    'Bio',
    'molscrub',
    'py3Dmol',
    'sklearn',
]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'rdkit': ('https://www.rdkit.org/docs/', None),
}

# myst-nb settings
nb_execution_mode = 'off'  # Don't execute notebooks during build
