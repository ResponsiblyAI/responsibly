Ethically
=========

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

**Toolkit for Auditing and Mitigating Bias and Fairness**
**of Machine Learning Systems 🔎🤖🔧**

*Ethically* is developed for **practitioners** and **researchers** in mind,
but also for learners. Therefore, it is compatible with
data science and machine learning tools of trade in Python,
such as Numpy, Pandas, and especially **scikit-learn**.

The primary goal is to be one-shop-stop for **auditing** bias
and fairness of machine learning systems, and the secondary one
is to mitigate bias and adjust fairness through **interventions**.
Besides, there is a particular focus on **NLP** models.

*Ethically* consists of three sub-packages:

1. ``ethically.dataset``
     Collection of common benchmark datasets from fairness research.

2. ``ethically.fairness``
     Demographic fairness in binary classification,
     including metrics and interventions.

3. ``ethically.we``
     Metrics and debiasing methods for bias (such as gender and race)
     in words embedding.

For fairness, *Ethically*'s functionality is aligned with the book
`Fairness and Machine Learning
- Limitations and Opportunities <https://fairmlbook.org>`_
by Solon Barocas, Moritz Hardt and Arvind Narayanan.

If you would like to ask for a feature or report a bug,
please open a
`new issue <https://github.com/EthicallyAI/ethically/issues/new>`_
or write us in `Gitter <https://gitter.im/EthicallyAI/ethically>`_.

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

Citation
--------

If you have used *Ethically* in a scientific publication,
we would appreciate citations to the following:

::

  @Misc{,
    author =    {Shlomi Hod},
    title =     {{Ethically}: Toolkit for Auditing and Mitigating Bias and Fairness of Machine Learning Systems},
    year =      {2018--},
    url = "http://docs.ethically.ai/",
    note = {[Online; accessed <today>]}
  }
