
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('..'))


project = 'shepherd-score'
copyright = f'2024-{datetime.now().year}, Kento Abeywardane'
author = 'Kento Abeywardane'

from shepherd_score import __version__  # noqa: E402
release = __version__


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

templates_path = []
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

master_doc = 'index'


html_theme = 'sphinx_book_theme'
pygments_style = 'sphinx'

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

copybutton_exclude = '.linenos, .gp'


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
napoleon_attr_annotations = False

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}
autodoc_typehints = 'description'
autodoc_mock_imports = [
    'torch',
    'numba',
    'triton',
    'jax',
    'jaxlib',
    'optax',
    'open3d',
    'meeko',
    'vina',
    'openbabel',
    'prolif',
    'biopython',
    'Bio',
    'MDAnalysis',
    'molscrub',
    'py3Dmol',
    'sklearn',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'rdkit': ('https://www.rdkit.org/docs/', None),
}

nb_execution_mode = 'off'  # Don't execute notebooks during build

myst_enable_extensions = ["dollarmath", "amsmath"]

def copy_tutorials(app):
    from pathlib import Path
    import shutil
    docs = Path(app.srcdir)
    for source in (docs.parent / "examples").glob("*.ipynb"):
        target = docs / "tutorials" / source.name
        if target.is_symlink():
            continue
        shutil.copyfile(source, target)


def setup(app):
    app.connect("builder-inited", copy_tutorials)
