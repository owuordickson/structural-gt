# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
import time

# import sphinx_gallery.gen_rst
# from furo.gen_tutorials import generate_tutorials

# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'StructuralGT'
copyright = f"{time.localtime().tm_year} Dickson Owuor"
author = 'Dickson Owuor'

# The full version, including alpha/beta/rc tags
release = '2.0'
version = '2.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# -- General configuration
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "myst_parser",
    #"furo.gen_tutorials",
    #"sphinx_gallery.gen_gallery",
    #"sphinx_github_changelog",
]


# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Napoleon settings
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
# See https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [("Returns", "params_style")]

# Autodoc
autoclass_content = "both"
autodoc_preserve_defaults = True


# This function removes the content before the parameters in the __init__ function.
# This content is often not useful for the website documentation as it replicates
# the class docstring.
def remove_lines_before_parameters(app, what, name, obj, options, lines):
    if what == "class":
        # ":param" represents args values
        first_idx_to_keep = next(
            (i for i, line in enumerate(lines) if line.startswith(":param")), 0
        )
        lines[:] = lines[first_idx_to_keep:]


def setup(app):
    app.connect("autodoc-process-docstring", remove_lines_before_parameters)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "StructuralGT Documentation"
html_baseurl = "https://structural-gt.readthedocs.io"
html_copy_source = False
# html_favicon = "_static/img/favicon.png"
html_theme_options = {
    #"light_logo": "img/sgt_black.svg",
    #"dark_logo": "img/sgt_white.svg",
    #"gtag": "G-6H9C8TWXZ8",
    "description": "A software tool that allows graph theory analysis of nano-structures.",
    #"image": "img/sgt-github.png",
    "versioning": True,
    "source_repository": "https://github.com/owuordickson/structural-gt",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = []


# -- Options for EPUB output
# epub_show_urls = 'footnote'