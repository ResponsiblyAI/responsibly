Ethically
=========
*in active development*


.. image:: https://img.shields.io/gitter/room/nwjs/nw.js.svg
   :alt: Join the chat at https://gitter.im/ethicallyAI/ethically
   :target: https://gitter.im/ethicallyAI/ethically?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. image:: https://img.shields.io/travis/ethicallyAI/ethically/master.svg
    :target: https://travis-ci.org/ethicallyAI/ethically

.. image:: https://img.shields.io/appveyor/ci/shlomihod/ethically/master.svg
   :target: https://ci.appveyor.com/project/shlomihod/ethically

.. image::  https://img.shields.io/coveralls/ethicallyAI/ethically/master.svg
   :target: hhttps://coveralls.io/r/ethicallyAI/ethically

.. image::  https://img.shields.io/scrutinizer/g/ethicallyAI/ethically.svg
  :target: https://scrutinizer-ci.com/g/ethicallyAI/ethically/?branch=master

.. image::  https://img.shields.io/pypi/v/ethically.svg
  :target: https://pypi.org/project/ethically


Toolbox for Auditing, Designing and Adjusting the Ethics of AI
Systems.


Auditing
  Mesuring the ethics of a trained model.
  For example, binary classification fariness mesures
  (demographic parity, ...), WEAT score of a words embedding.

Designing (Pre-Mitigation)
  Improving the ethics of a model or data before or while the training.
  For example, fairness regularization, data preprocessing.

Adjusting (Post-Mitigation)
  Improving the ethics of a trained model.
  For example, gender debiasing of a words embedding.


Usage
-----

After installation, the package can imported:

.. code:: sh

   $ python
   >>> from ethically.we import GenderBiasWE
   >>> from gensim import downloader
   >>> w2v_model = downloader.load('word2vec-google-news-300')
   >>> w2v_gender_bias_we = GenderBiasWE(w2v_model)
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


Development Roadmap - 2018
--------------------------
1. Words Embedding

   1. Bolukbasi et al. Debiasing (Gender & Race)
   2. Word Embedding Association Test (WEAT)

2. Fairness in Binary Classification

   1. Loading of common datasets in ML fairness research
   2. Auditing (e.g demographic parity)
   3. Designing (e.g. fairness regularization)
   4. Adjusting (e.g.  reject option classification)

3. Web Auditing Tool (for non-programmers)
