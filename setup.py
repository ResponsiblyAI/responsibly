#!/usr/bin/env python

import os
import sys

import setuptools


PACKAGE_NAME = 'responsibly'


def read_package_variable(key, filename='__init__.py'):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, filename)
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ', 2)
            if parts[:-1] == [key, '=']:
                return parts[-1].strip("'")
    sys.exit("'%s' not found in '%s'", key, module_path)


def build_description():
    """Build a description for the project from documentation files."""
    with open("README.rst") as f:
        readme = f.read()
    with open("CHANGELOG.rst") as f:
        changelog = f.read()
    return readme + '\n' + changelog


setuptools.setup(
    name=read_package_variable('__project__'),
    version=read_package_variable('__version__'),

    description=read_package_variable('__description__'),
    url=read_package_variable('__url__'),
    author=read_package_variable('__author__'),
    author_email=read_package_variable('__author_email__'),

    packages=setuptools.find_packages(),

    include_package_data=True,

    # entry_points={'console_scripts': [
    #     'responsibly-cli = responsibly.cli:main',
    # ]},

    long_description=build_description(),
    long_description_content_type='text/x-rst',
    license=read_package_variable('__license__'),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    python_requires='>=3.6, <3.9',

    install_requires=[
        "numpy >= 1.15",
        "scipy >= 1.1",
        "pandas >= 0.23",
        "matplotlib >= 2.2, < 3",
        "seaborn >= 0.9",
        "scikit-learn >= 0.19",
        "gensim >= 3.7, < 3.8",
        "tabulate >= 0.8",
        "six >= 1.10",
        "click >= 6.0",
        "tqdm >= 4.24",
        "mlxtend >= 0.13, < 0.17",
    ],
)
