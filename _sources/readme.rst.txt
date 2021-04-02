Responsibly
===========

.. image:: https://img.shields.io/badge/docs-passing-brightgreen.svg
    :target: https://docs.responsibly.ai

.. image:: https://img.shields.io/gitter/room/nwjs/nw.js.svg
   :alt: Join the chat at https://gitter.im/ResponsiblyAI/responsibly
   :target: https://gitter.im/ResponsiblyAI/responsibly

.. image:: https://img.shields.io/github/workflow/status/ResponsiblyAI/responsibly/CI/master.svg
    :target: https://github.com/ResponsiblyAI/responsibly/actions/workflows/ci.yml
 
.. image::  https://img.shields.io/coveralls/ResponsiblyAI/responsibly/master.svg
   :target: https://coveralls.io/r/ResponsiblyAI/responsibly

.. image::  https://img.shields.io/scrutinizer/g/ResponsiblyAI/responsibly.svg
  :target: https://scrutinizer-ci.com/g/ResponsiblyAI/responsibly/?branch=master

.. image::  https://img.shields.io/pypi/v/responsibly.svg
  :target: https://pypi.org/project/responsibly

.. image::  https://img.shields.io/github/license/ResponsiblyAI/responsibly.svg
    :target: https://docs.responsibly.ai/about/license.html

**Toolkit for Auditing and Mitigating Bias and Fairness**
**of Machine Learning Systems 🔎🤖🧰**

*Responsibly* is developed for **practitioners** and **researchers** in mind,
but also for learners. Therefore, it is compatible with
data science and machine learning tools of trade in Python,
such as Numpy, Pandas, and especially **scikit-learn**.

The primary goal is to be one-shop-stop for **auditing** bias
and fairness of machine learning systems, and the secondary one
is to mitigate bias and adjust fairness through
**algorithmic interventions**.
Besides, there is a particular focus on **NLP** models.

*Responsibly* consists of three sub-packages:

1. ``responsibly.dataset``
     Collection of common benchmark datasets from fairness research.

2. ``responsibly.fairness``
     Demographic fairness in binary classification,
     including metrics and algorithmic interventions.

3. ``responsibly.we``
     Metrics and debiasing methods for bias (such as gender and race)
     in word embedding.

For fairness, *Responsibly*'s functionality is aligned with the book
`Fairness and Machine Learning
- Limitations and Opportunities <https://fairmlbook.org>`_
by Solon Barocas, Moritz Hardt and Arvind Narayanan.

If you would like to ask for a feature or report a bug,
please open a
`new issue <https://github.com/ResponsiblyAI/responsibly/issues/new>`_
or write us in `Gitter <https://gitter.im/ResponsiblyAI/responsibly>`_.

Requirements
------------

-  Python 3.6+

Installation
------------

Install responsibly with pip:

.. code:: sh

   $ pip install responsibly

or directly from the source code:

.. code:: sh

   $ git clone https://github.com/ResponsiblyAI/responsibly.git
   $ cd responsibly
   $ python setup.py install

Citation
--------

If you have used *Responsibly* in a scientific publication,
we would appreciate citations to the following:

::

  @Misc{,
    author = {Shlomi Hod},
    title =  {{Responsibly}: Toolkit for Auditing and Mitigating Bias and Fairness of Machine Learning Systems},
    year =   {2018--},
    url =    "http://docs.responsibly.ai/",
    note =   {[Online; accessed <today>]}
  }
