Unix: |Unix| Windows: |Windows|

Metrics: |Coverage| |Code Quality|

Usage: |PyPI|


.. |Unix| image:: https://img.shields.io/travis/ethicallyAI/ethically/master.svg
.. Unix: https://travis-ci.org/ethicallyAI/ethically
.. |Windows| image:: https://img.shields.io/appveyor/ci/shlomihod/ethically/master.svg
.. Windows: https://ci.appveyor.com/project/shlomihod/ethically
.. |Coverage| image:: https://img.shields.io/coveralls/ethicallyAI/ethically/master.svg
.. Coverage: https://coveralls.io/r/ethicallyAI/ethically
.. |Code Quality| image:: https://img.shields.io/scrutinizer/g/ethicallyAI/ethically.svg
.. Code Quality: https://scrutinizer-ci.com/g/ethicallyAI/ethically/?branch=master
.. |PyPI| image:: https://img.shields.io/pypi/v/ethically.svg
.. PyPI: https://pypi.org/project/ethically

Ethically
=========
*in active development*

Python Package for Auditing, Designing and Adjusting the Ethics of AI
Systems.

.. glossary::

   Auditing
      Mesuring the ethics of a trained model.
      For example, binary classification fariness mesures
      (demographic parity, ...).

   Designing
      Improving the ethics of a model before training.
      For example, fairness fairness regularization.

   Adjusting
      Improving the ethics of trained model.
      For example, gender debiasing of words embedding.


Usage
-----

After installation, the package can imported:

.. code:: sh

   $ python
   >>> import ethically
   >>> from gensim import downloader
   >>> w2v_model = downloader.load('word2vec-google-news-300')
   >>> w2v_gender_bias_we = ethically.we.GenderBiasWE(w2v_model)
   >>> w2v_gender_bias_we.calc_direct_bias()
   0.07307904249481942
   >>> w2v_gender_bias_we.debias()
   >>> w2v_gender_bias_we.calc_direct_bias()
   1.7964246601064155e-09

Requirements
------------

-  Python 3.5+

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
