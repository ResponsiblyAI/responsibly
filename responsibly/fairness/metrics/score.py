from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils.multiclass import unique_labels

from responsibly.fairness.metrics.utils import (
    _assert_binary, _groupby_y_x_sens,
)


def _proportion(data, labels):
    counts = Counter(data)
    assert set(counts.keys()).issubset(labels)
    return (counts[labels[1]]
            / (counts[labels[0]] + counts[labels[1]]))


def _get_labels(ys, labels):

    if labels is None:
        labels = unique_labels(ys)
    else:
        labels = np.asarray(labels)
        if np.all([label not in ys for label in labels]):
            raise ValueError('At least one label specified must be in y.')

    return labels


def _normalize_by_attr(y_score, x_sens, ndigits=1):
    y_score_within = y_score[:]

    for indices in x_sens.groupby(x_sens).groups.values():
        y_score_within[indices] = (y_score_within[indices]
                                   .rank(pct=True))

    y_score_within = (np.floor(y_score_within * (10**ndigits))
                      / (10**ndigits))

    return y_score_within


def independence_score(y_score, x_sens,
                       as_df=False):
    """Compute the independence criteria for score prediction.

    In classification terminology, it is the **acceptance rate**
    grouped by the score and the sensitive attribute.

    :param y_score: Estimated target score as returned by a classifier.
    :param x_sens: Sensitive attribute values corresponded to each
                   estimated target.
    :param as_df: Whether to return the results as ``dict`` (if ``False``)
                  or as :class:`pandas.DataFrame`(if ``True``).
    :return: Independence criteria.
    :rtype: dict or :class:`pandas.DataFrame`
    """
    criterion = pd.crosstab(index=y_score,
                            columns=x_sens,
                            normalize='columns')

    if not as_df:
        criterion = criterion.to_dict()

    return criterion


def separation_score(y_true, y_score, x_sens,
                     labels=None,
                     as_df=False):
    """Compute the separation criteria for score prediction.

    In classification terminology, it is the **FPR** and **TPR**
    grouped by the score and the sensitive attribute.

    :param y_true: Binary ground truth (correct) target values.
    :param y_score: Estimated target score as returned by a classifier.
    :param x_sens: Sensitive attribute values corresponded to each
                   estimated target.
    :param as_df: Whether to return the results as ``dict`` (if ``False``)
                  or as :class:`pandas.DataFrame` (if ``True``).
    :return: Separation criteria.
    :rtype: dict or :class:`pandas.DataFrame`
    """

    _assert_binary(y_true)

    labels = _get_labels(y_score, labels)

    criterion = pd.crosstab(index=y_score,
                            columns=[y_true, x_sens],
                            normalize=True)

    if not as_df:
        criterion = criterion.to_dict()

    return criterion


def sufficiency_score(y_true, y_score, x_sens,
                      labels=None,
                      within_score_percentile=False,
                      as_df=False):
    """Compute the sufficiency criteria for score prediction.

    In classification terminology, it is the **PPV** and the **NPV**
    grouped by the score and the sensitive attribute.

    :param y_true: Binary ground truth (correct) target values.
    :param y_score: Estimated target score as returned by a classifier.
    :param x_sens: Sensitive attribute values corresponded to each
                   target.
    :param as_df: Whether to return the results as ``dict`` (if ``False``)
                  or as :class:`pandas.DataFrame` (if ``True``).
    :return: Sufficiency criteria.
    :rtype: dict or :class:`pandas.DataFrame`
    """

    _assert_binary(y_true)

    labels = _get_labels(y_true, labels)

    if within_score_percentile:
        y_score = _normalize_by_attr(y_score, x_sens,
                                     within_score_percentile)

    criterion = pd.crosstab(index=y_score,
                            columns=x_sens,
                            values=y_true,
                            aggfunc=partial(_proportion,
                                            labels=labels))

    if not as_df:
        criterion = criterion.to_dict()

    return criterion


def _all_equal(iterator):
    iterator = iter(iterator)

    try:
        first = next(iterator)
    except StopIteration:
        return True

    try:
        return all(np.allclose(first, rest) for rest in iterator)
    except ValueError:
        return False


def roc_curve_by_attr(y_true, y_score, x_sens,
                      pos_label=None, sample_weight=None,
                      drop_intermediate=False):
    """Compute Receiver operating characteristic (ROC) by attribute.

    Based on :func:`sklearn.metrics.roc_curve`

    :param y_true: Binary ground truth (correct) target values.
    :param y_score: Estimated target score as returned by a classifier.
    :param x_sens: Sensitive attribute values corresponded to each
                   estimated target.
    :param pos_label: Label considered as positive and others
                      are considered negative.
    :param sample_weight: Sample weights.
    :param drop_intermediate: Whether to drop some suboptimal
                              thresholds which would not appear on
                              a plotted ROC curve.
                              This is useful in order to create
                              lighter ROC curves.
    :return: For each value of sensitive attribute:
             - fpr - Increasing false positive rates such
               that element i is the false positive rate
               of predictions with score >= thresholds[i].
             - fpr - Increasing true positive rates such
               that element i is the true positive rate
               of predictions with score >= thresholds[i].
             - thresholds -
               Decreasing thresholds on the decision function
               used to compute fpr and tpr. thresholds[0] represents
               no instances being predicted and is arbitrarily set
               to max(y_score) + 1.
    :rtype: dict

    """

    grouped = _groupby_y_x_sens(y_true, y_score, x_sens)

    # pylint: disable=too-many-function-args
    roc_curves = {x_sens_value: roc_curve(group['y_true'],
                                          group['y_score'],
                                          pos_label, sample_weight,
                                          drop_intermediate)
                  for x_sens_value, group in grouped}

    if not _all_equal(thresholds
                      for _, _, thresholds in roc_curves.values()):
        raise NotImplementedError('All the scores values should'
                                  ' appear for each sensitive'
                                  ' attribute value.'
                                  ' It will be implemented'
                                  ' in the future.'
                                  ' Please post your use-case in'
                                  ' https://github.com/ResponsiblyAI/responsibly/issues/15')  # pylint: disable=line-too-long

    return roc_curves


def roc_auc_score_by_attr(y_true, y_score, x_sens,
                          sample_weight=None):
    """Compute Area Under the ROC (AUC) by attribute.

    Based on function:`sklearn.metrics.roc_auc_score`

    :param y_true: Binary ground truth (correct) target values.
    :param y_score: Estimated target score as returned by a classifier.
    :param x_sens: Sensitive attribute values corresponded to each
                   estimated target.
    :param sample_weight: Sample weights.
    :return: ROC AUC grouped by the sensitive attribute.
    :rtype: dict
    """

    grouped = _groupby_y_x_sens(y_true, y_score, x_sens)

    return {x_sens_value: roc_auc_score(group['y_true'],
                                        group['y_score'],
                                        sample_weight=sample_weight)
            for x_sens_value, group in grouped}
