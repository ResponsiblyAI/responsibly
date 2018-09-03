Ethically
=========
*in active development*


.. image:: https://img.shields.io/badge/docs-passing-brightgreen.svg
    :target: https://docs.ethically.ai

.. image:: https://img.shields.io/gitter/room/nwjs/nw.js.svg
   :alt: Join the chat at https://gitter.im/EthicallyAI/ethically
   :target: https://gitter.im/EthicallyAI/ethically

.. image:: https://img.shields.io/travis/EthicallyAI/ethically/master.svg
    :target: https://travis-ci.org/EthicallyAI/ethically

.. image:: https://img.shields.io/appveyor/ci/shlomihod/ethically/master.svg
   :target: https://ci.appveyor.com/project/shlomihod/ethically

.. image::  https://img.shields.io/coveralls/EthicallyAI/ethically/master.svg
   :target: https://coveralls.io/r/EthicallyAI/ethically

.. image::  https://img.shields.io/scrutinizer/g/EthicallyAI/ethically.svg
  :target: https://scrutinizer-ci.com/g/EthicallyAI/ethically/?branch=master

.. image::  https://img.shields.io/pypi/v/ethically.svg
  :target: https://pypi.org/project/ethically

.. image::  https://img.shields.io/github/license/EthicallyAI/ethically.svg
    :target: http://docs.ethically.ai/about/license.html

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

After installation, the package can be imported:

.. code:: sh

   $ python
   >>> from ethically.we import GenderBiasWE
   >>> from ethically.we.data import load_w2v_small
   >>> w2v_small_model = load_w2v_small()
   >>> w2v_gender_bias_we = GenderBiasWE(w2v_small_model, only_lower=True)
   >>> w2v_gender_bias_we.calc_direct_bias()
   0.07307905390764366
   >>> w2v_gender_bias_we.debias()
   >>> w2v_gender_bias_we.calc_direct_bias()
   1.8275163172819303e-09

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

   $ git clone https://github.com/EthicallyAI/ethically.git
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
