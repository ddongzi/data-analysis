# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DA'
copyright = '2026, 蓝色夕阳'
author = '蓝色夕阳'

html_favicon = "_static/favicon.ico"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",       # 这是解析 Jupyter Notebook 的核心插件
    # ... 其他插件
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.md': 'myst-nb',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_theme_options = {
    "github_url": "https://github.com/ddongzi/data-analysis",

    "show_nav_level": 0, # 顶部导肮只显示caption
    "show_toc_level": 2, # 每页侧边栏显示2级
    "collapse_navigation": False # 全展开
}

# 补充自定义css
html_css_files = [
    'css/custom.css',
]