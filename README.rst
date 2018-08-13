Unix:

.. image:: https://img.shields.io/travis/ethicallyAI/ethically/master.svg

    :target: https://travis-ci.org/ethicallyAI/ethically
    
Usage:

.. image:: https://img.shields.io/pypi/v/ethically.svg

    :target: https://pypi.org/project/ethically


Ethically
=========

Python Package for Designing, Auditing and Adjusting the Ethics of AI
Systems.

Features (planned, incomplete list)
-----------------------------------

-  [ ] Words Embedding

   -  [ ] Bolukbasi, Tolga, Kai-Wei Chang, James Y. Zou, Venkatesh
      Saligrama, and Adam T. Kalai. `Man is to computer programmer as
      woman is to homemaker? debiasing word
      embeddings <https://arxiv.org/abs/1607.06520>`__. NIPS.
   -  [ ] Caliskan, Aylin, Joanna J. Bryson, and Arvind Narayanan. 2017.
      `Semantics derived automatically from language corpora contain
      human-like
      biases <https://researchportal.bath.ac.uk/en/publications/semantics-derived-automatically-from-language-corpora-necessarily>`__.
      Science.

-  [ ] Fairness in Classification

   -  [ ] Designing - Regularization
   -  [ ] Auditing - Metrics
   -  [ ] Adjusting

Requirements
------------

-  Python 3.6+

Installation
------------

Install ethically with pip:

.. code:: sh

   $ pip install ethically

or directly from the source code:

.. code:: sh

   $ git clone https://github.com/ethicallyAI/ethically.git
   $ cd ethically
   $ python setup.py install

Usage
-----

After installation, the package can imported:

.. code:: sh

   $ python
   >>> import ethically
   >>> ethically.__version__

Credits
-------

-  ``ethically.we.tolga`` code and data is based on
   https://github.com/tolga-b/debiaswe
