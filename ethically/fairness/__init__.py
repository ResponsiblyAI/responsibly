"""
Demographic Fairness in Binary Classification.

The common and simples setting, but not the one, of fairness
of a binary classifier is the demographic one.
It is assume that there is one sensitive attribute or more
that represents one or more demographic groups
(e.g., by gender, race or age), for which a classifier
should be fair.

.. important::
    The terminology and functionality is aligned with the
    book `Fairness and Machine Learning
    - Limitations and Opportunities <https://fairmlbook.org>`_
    by Solon Barocas, Moritz Hardt and Arvind Narayanan.
    Therefore, it is advised to get familiar with
    `Chapter 2 <https://fairmlbook.org/demographic.html>`_,
    as it summarized the current core knowledge regarding fairness
    in classification.

Currently, the :mod:`ethically.fairness` module has two components:

1. **Metrics** (:mod:`ethically.fairness.metrics`) for measuring unfairness.
2. **Interventions** (:mod:`ethically.fairness.interventions`) for satisfying
   fairness criteria.

"""
