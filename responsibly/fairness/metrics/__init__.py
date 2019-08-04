"""
Demographic Classification Fairness Criteria.

The objectives of the demographic classification fairness criteria
is to measure unfairness towards sensitive attribute valuse.

The metrics have the same interface and behavior as the ones in
:mod:`sklearn.metrics`
(e.g., using ``y_true``, ``y_pred`` and ``y_score``).

One should keep in mind that the criteria are intended
to *measure unfairness, rather than to prove fairness*, as it stated in
the paper `Equality of opportunity in supervised learning <https://arxiv.org/abs/1610.02413>`_
by Hardt et al. (2016):

    ... satisfying [the demographic criteria] should not be
    considered a conclusive proof of fairness.
    Similarly, violations of our condition are not meant
    to be a proof of unfairness.
    Rather we envision our framework as providing a reasonable way
    of discovering and measuring potential concerns that require
    further scrutiny. We believe that resolving fairness concerns is
    ultimately impossible without substantial domain-specific
    investigation.

The output of binary classifiers can come in two forms, either giving
a binary outcome prediction for input or producing
a real number score, which the common one is the probability
for the positive or negative label
(such as the method ``proba`` of an ``Estimator`` in ``sklearn``).
Therefore, the criteria come in two flavors, one for **binary** output,
and the second for **score** output.

The fundamental concept for defining the fairness criteria
is `conditional independence <https://en.wikipedia.org/wiki/Conditional_independence>`_.
Using *Machine Learning and Fairness* book's notions:

- ``A`` - Sensitive attribute
- ``Y`` - Binary ground truth (correct) target
- ``R`` - Estimated binary targets or score as returned by a classifier

There are three demographic fairness criteria for classification:

1. Independence - R⊥A

2. Separation - R⊥A∣Y

3. Sufficiency - Y⊥A∣R

"""


from responsibly.fairness.metrics.binary import (
    independence_binary, report_binary, separation_binary, sufficiency_binary,
)
from responsibly.fairness.metrics.score import (
    independence_score, roc_auc_score_by_attr, roc_curve_by_attr,
    separation_score, sufficiency_score,
)
from responsibly.fairness.metrics.visualization import (
    distplot_by, plot_roc_by_attr, plot_roc_curves,
)
