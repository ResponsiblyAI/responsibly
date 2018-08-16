Unix: [![Unix Build Status](https://img.shields.io/travis/ethicallyAI/ethically/master.svg)](https://travis-ci.org/ethicallyAI/ethically) Windows: [![Windows Build Status](https://img.shields.io/appveyor/ci/ethicallyAI/ethically/master.svg)](https://ci.appveyor.com/project/shlomihod/ethically)<br> Metrics: [![Coverage Status](https://img.shields.io/coveralls/ethicallyAI/ethically/master.svg)](https://coveralls.io/r/ethicallyAI/ethically) [![Scrutinizer Code Quality](https://img.shields.io/scrutinizer/g/ethicallyAI/ethically.svg)](https://scrutinizer-ci.com/g/ethicallyAI/ethically/?branch=master)<br>Usage: [![PyPI Version](https://img.shields.io/pypi/v/ethically.svg)](https://pypi.org/project/ethically)

# Ethically
Python Package for Designing, Auditing and Adjusting the Ethics of AI Systems.

## Features (planned, incomplete list)
- [ ] Words Embedding
  - [ ] Bolukbasi Tolga, Kai-Wei Chang, James Y. Zou, Venkatesh Saligrama, and Adam T. Kalai. [Man is to computer programmer as woman is to homemaker? debiasing word embeddings](https://arxiv.org/abs/1607.06520). NIPS.
  - [ ] Caliskan, Aylin, Joanna J. Bryson, and Arvind Narayanan. 2017. [Semantics derived automatically from language corpora contain human-like biases](https://researchportal.bath.ac.uk/en/publications/semantics-derived-automatically-from-language-corpora-necessarily). Science.

- [ ] Fairness in Classification
  - [ ] Designing - Regularization
  - [ ] Auditing - Metrics
  - [ ] Adjusting

## Requirements

* Python 3.5+

## Installation

Install ethically with pip:

```sh
$ pip install ethically
```

or directly from the source code:

```sh
$ git clone https://github.com/ethicallyAI/ethically.git
$ cd ethically
$ python setup.py install
```

## Usage

After installation, the package can imported:

```sh
$ python
>>> import ethically
>>> ethically.__version__
```


## Credits
* `ethically.we.core` code and data is based on
https://github.com/tolga-b/debiaswe
