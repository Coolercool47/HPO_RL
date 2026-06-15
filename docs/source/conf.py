# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Добавляем корень проекта в путь для импорта hpo_rl
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

project = 'HPO_RL'
copyright = '2026, Петерс Е. А., Матков Н. К., Тимошин Э. К.'
author = 'Петерс Е. А., Матков Н. К., Тимошин Э. К. (рук. Парфенов Д. В.)'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # авто документации из docstrings
    'sphinx.ext.viewcode',  # ссылки на исходный код
    'sphinx.ext.napoleon',  # поддержка Google и NumPy стиля документации
    'sphinx.ext.todo',  # поддержка TODO
    'sphinx.ext.coverage',  # проверяет покрытие документации
    'sphinx.ext.ifconfig',  # условные операторы
    'sphinx.ext.autosummary',  # генерация заглушек
    'sphinx.ext.mathjax',  # рендеринг математических формул
    'sphinx_autodoc_typehints',  # подсказки по типам данных
    ]

autosummary_generate = True

autodoc_mock_imports = [
    "matplotlib",
    "mpl_toolkits",
    "pandas",
    "scipy",
    "torch",
    "torchvision",
    "gymnasium",
    "stable_baselines3",
    "sb3_contrib",
    "tianshou",
    "tqdm",
    "wandb",
    "yaml",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Исключаем служебные атрибуты ABC и другие внутренние атрибуты
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'show-inheritance': True,
    'exclude-members': '_abc_impl,__weakref__,__dict__,__module__,__doc__,__annotations__,FUNCTIONS,HYPERPARAMETERS'
}

templates_path = ['_templates']
exclude_patterns = []

# Подавляем предупреждения о generated файлах, которые не включены в toctree
suppress_warnings = ['toc.not_included']

language = 'ru'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_css_files = ['custom.css']

html_theme_options = {
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Настройка MathJax 3 для корректного отображения формул
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
# MathJax 3 автоматически обрабатывает формулы из Sphinx, дополнительная конфигурация не требуется
