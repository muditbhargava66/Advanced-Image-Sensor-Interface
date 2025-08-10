# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------

project = 'Advanced Image Sensor Interface'
current_year = datetime.datetime.now().year
copyright = f'{current_year}, Mudit Bhargava'
author = 'Mudit Bhargava'

# The full version, including alpha/beta/rc tags
release = 'v2.0.0'
version = '2.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'myst_parser',
]

# Try to import optional extensions
try:
    import sphinx_autodoc_typehints
    extensions.append('sphinx_autodoc_typehints')
except ImportError:
    pass

try:
    import sphinx_copybutton
    extensions.append('sphinx_copybutton')
except ImportError:
    pass

try:
    import sphinxcontrib.mermaid
    extensions.append('sphinxcontrib.mermaid')
except ImportError:
    pass

try:
    import sphinx_design
    extensions.append('sphinx_design')
except ImportError:
    pass

try:
    import nbsphinx
    extensions.append('nbsphinx')
except ImportError:
    pass

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Additional settings for myst-parser
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Extension configuration -------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
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

# Intersphinx mapping (only include reliable sources)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
    'scikit-image': ('https://scikit-image.org/docs/stable/', None),
}

# MyST parser settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# MyST heading anchors
myst_heading_anchors = 3

# MyST URL schemes
myst_url_schemes = ["http", "https", "mailto"]

# Copy button settings (if available)
if 'sphinx_copybutton' in extensions:
    copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
    copybutton_prompt_is_regexp = True

# HTML theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add custom CSS
html_css_files = [
    'custom.css',
]

# Todo extension settings
todo_include_todos = True

# Coverage extension settings
coverage_show_missing_items = True
