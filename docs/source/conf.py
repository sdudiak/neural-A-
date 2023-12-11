# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# sys.path.insert(0, os.path.abspath("/workspaces/neural-Astar/*"))
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Neural Astar"
copyright = "2023, Stanisław Dudiak"
author = "Stanisław Dudiak"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

strip_signature_backslash = False
napoleon_strip_signature_backslash = False
autodoc_strip_signature_backslash = False
strip_signature_backslash = False
