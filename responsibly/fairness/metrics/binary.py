import pandas as pd
from pandas.core.algorithms import unique as _unique
from sklearn.metrics import confusion_matrix

from responsibly.fairness.metrics.utils import _assert_binary


def _select_dict(d, keys):
    return {k: d[k] for k in keys}


def _nested_select_dict(d, nested_keys):
    return {k:
            _select_dict(v, nested_keys)
            for k, v in d.items()}


def _choose_other(item, iterable):
    return next(other for other in iterable
                if other != item)


def _nested_diff_and_ratio(d, nested_key, first, second):

    assert d.keys() == {first, second}

    return {'diff': d[first][nested_key] - d[second][nested_key],
            'ratio': d[first][nested_key] / d[second][nested_key]}


def binary_stats_by_attr(y_true, y_pred, x_attr,
                         labels=None):
    # pylint: disable=too-many-locals

    _assert_binary(y_true, y_pred)

    stats = {}

    for x_att_val in _unique(x_attr):
        mask = (x_attr == x_att_val)

        tn, fp, fn, tp = confusion_matrix(y_true[mask],
                                          y_pred[mask],
                                          labels=labels).ravel()

        pos = tp + fn
        neg = tn + fp

        acceptance = tp + fp
        rejection = tn + fn

        correct = tp + tn

        total = pos + neg

        stats[x_att_val] = {
            'total': int(total),
            'proportion': total / len(x_attr),
            'pos': int(pos),
            'neg': int(neg),
            'base_rate': pos / total,
            'acceptance_rate': acceptance / total,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'accuracy': correct / total,
            'balanced_accuracy': (tp / pos + tn / neg) / 2,
            'tpr': tp / pos,
            'tnr': tn / neg,
            'fnr': fn / pos,
            'fpr': fp / neg,
            'ppv': tp / acceptance,
            'npv': tn / rejection
        }

    return stats


def compare_privileged(stats,
                       x_sens_privileged=None):
    # pylint: disable=line-too-long

    if len(stats) != 2:
        if x_sens_privileged is not None:
            raise ValueError('x_sens_privileged should have'
                             'only two values for comparision'
                             '(difference and ratio).')

        return None

    comparison = {}

    if x_sens_privileged is None:
        x_sens_privileged = next(iter(stats))

    x_sens_unprivileged = _choose_other(x_sens_privileged,
                                        stats)

    comparison['x_sens_privileged'] = x_sens_privileged
    comparison['x_sens_unprivileged'] = x_sens_unprivileged

    comparison['metrics'] = {}

    metrics = next(iter(stats.values())).keys()

    for metric in metrics:
        comparison['metrics'][metric] = _nested_diff_and_ratio(stats,
                                                               metric,
                                                               x_sens_unprivileged,
                                                               x_sens_privileged)

    return comparison


def group_fairness_criterion_binary(y_true, y_pred, x_sens,
                                    metrics,
                                    x_sens_privileged=None,
                                    labels=None,
                                    as_df=False):

    stats = binary_stats_by_attr(y_true, y_pred, x_sens,
                                 labels=labels)

    criterion = _nested_select_dict(stats,
                                    metrics)

    comparison = compare_privileged(criterion,
                                    x_sens_privileged)

    if as_df:
        criterion = pd.DataFrame(criterion)

        if comparison is not None:
            vs_name = ('{x_sens_unprivileged} vs. {x_sens_privileged}'
                       .format(**comparison))

            comparison = pd.DataFrame(comparison['metrics'])
            comparison.index.name = vs_name

    return criterion, comparison


def independence_binary(y_pred, x_sens,
                        x_sens_privileged=None,
                        labels=None,
                        as_df=False):
    """Compute the independence criteria for binary prediction.

    In classification terminology, it is the **acceptance rate**
    grouped by the sensitive attribute.

    :param y_pred: Estimated targets as returned by a classifier.
    :param x_sens: Sensitive attribute values corresponded to each
                   target.
    :param x_sens_privileged: The privileged value in the
                              sensitive attribute. Relevent only
                              if there are only two values for
                              the sensitive attribute.
    :param labels: List of labels to choose the negative and positive target.
                   This may be used to reorder or select a subset of labels.
                   If none is given, those that appear at least once in
                   y_pred are used in sorted order; first is negative
                   and the second is positive.
    :param as_df: Whether to return the results as `dict` (if `False`)
                  or as :class:`pandas.DataFrame` (if `True`).
    :return: Independence criteria and comparision if there are
             only two values for the sensitive attribute.
    :rtype: tuple
    """

    # hack to keep the same strutcure of code
    # for independence as seperation and sufficiency
    # we take only acceptance_rate
    return group_fairness_criterion_binary(y_pred, y_pred, x_sens,
                                           ('acceptance_rate',),
                                           x_sens_privileged,
                                           labels,
                                           as_df)


def separation_binary(y_true, y_pred, x_sens,
                      x_sens_privileged=None,
                      labels=None,
                      as_df=False):
    """Compute the separation criteria for binary prediction.

    In classification terminology, it is the **TPR**, **FPR**,
    **TNR** and **FNR** grouped by the sensitive attribute.

    :param y_true: Binary ground truth (correct) target values.
    :param y_pred: Estimated binary targets as returned
                   by a classifier.
    :param x_sens: Sensitive attribute values corresponded to each
                   target.
    :param x_sens_privileged: The privileged value in the
                              sensitive attribute. Relevent only
                              if there are only two values for
                              the sensitive attribute.
    :param labels: List of labels to choose the negative and positive target.
                   This may be used to reorder or select a subset of labels.
                   If none is given, those that appear at least once in
                   y_pred are used in sorted order; first is negative
                   and the second is positive.
    :param as_df: Whether to return the results as `dict` (if `False`)
                  or as :class:`pandas.DataFrame` (if `True`).
    :return: Separation criteria and comparision if there are
             only two values for the sensitive attribute.
    :rtype: tuple
    """

    return group_fairness_criterion_binary(y_true, y_pred, x_sens,
                                           ('tpr', 'fpr', 'tnr', 'fnr'),
                                           x_sens_privileged,
                                           labels,
                                           as_df)


def sufficiency_binary(y_true, y_pred, x_sens,
                       x_sens_privileged=None,
                       labels=None,
                       as_df=False):
    """Compute the sufficiency criteria for binary prediction.

    In classification terminology, it is the **PPV** and **NPV**
    grouped by the sensitive attribute.

    :param y_true: Binary ground truth (correct) target values.
    :param y_pred: Binary estimated targets as returned by
                   a classifier.
    :param x_sens: Sensitive attribute values corresponded to each
                   target.
    :param x_sens_privileged: The privileged value in the
                              sensitive attribute. Relevent only
                              if there are only two values for
                              the sensitive attribute.
    :param labels: List of labels to choose the negative and positive target.
                   This may be used to reorder or select a subset of labels.
                   If none is given, those that appear at least once in
                   y_pred are used in sorted order; first is negative
                   and the second is positive.
    :param as_df: Whether to return the results as `dict` (if `False`)
                  or as :class:`pandas.DataFrame` (if `True`).
    :return: Sufficiency criteria and comparision if there are
             only two values for the sensitive attribute.
    :rtype: tuple
    """

    return group_fairness_criterion_binary(y_true, y_pred, x_sens,
                                           ('ppv', 'npv'),
                                           x_sens_privileged,
                                           labels,
                                           as_df)


def report_binary(y_true, y_pred, x_sens,
                  labels=None):
    """Generate a report of criteria for binary prediction.

    In classification terminology, the statistics are
    grouped by the sensitive attribute:
    - Number of observations per group
    - Proportion of of observations per group
    - Base rate
    - Acceptance rate
    - FNR
    - TPR
    - PPV
    - NPV

    :param y_true: Binary ground truth (correct) target values.
    :param y_pred: Binary estimated targets as returned by
                   a classifier.
    :param x_sens: Sensitive attribute values corresponded to each
                   target.
    :param labels: List of labels to choose the negative and positive target.
                   This may be used to reorder or select a subset of labels.
                   If none is given, those that appear at least once in
                   y_pred are used in sorted order; first is negative
                   and the second is positive.
    :return: Classification statistics grouped by the
             sensitive attribute.
    :rtype: :class:`pandas.DataFrame`
    """

    stats = binary_stats_by_attr(y_true, y_pred, x_sens, labels)
    stats_df = pd.DataFrame(stats)

    return stats_df.loc[['total', 'proportion', 'base_rate',
                         'acceptance_rate', 'accuracy',
                         'fnr', 'fpr', 'ppv', 'npv']]
