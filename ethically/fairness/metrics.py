"""
Fairness Demographic Classification Criteria.

Based on:
https://fairmlbook.org/demographic.html
"""

import pandas as pd
from pandas.core.algorithms import unique as _unique
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import _check_targets


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


def _assert_binary(y1, y2=None):

    if y2 is None:
        y2 = y1

    y_type, _, _ = _check_targets(y1, y2)

    if y_type != 'binary':
        raise ValueError('y_true and y_pred must be binary.')


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

        p = tp + fn
        n = tn + fp

        acceptance = tp + fp
        rejection = tn + fn

        correct = tp + tn

        total = p + n

        stats[x_att_val] = {
            'total': total,
            'p': p,
            'n': n,
            'base_rate': p / total,
            'acceptance_rate': acceptance / total,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'accuracy': correct / total,
            'balanced_accuracy': (tp/p + tn/n)/2,
            'tpr': tp / p,
            'tnr': tn / n,
            'fnr': fn / p,
            'fpr': fp / n,
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

    return group_fairness_criterion_binary(y_true, y_pred, x_sens,
                                           ('tpr', 'fpr', 'tnr', 'fnr'),
                                           x_sens_privileged,
                                           labels,
                                           as_df)


def sufficiency_binary(y_true, y_pred, x_sens,
                       x_sens_privileged=None,
                       labels=None,
                       as_df=False):

    return group_fairness_criterion_binary(y_true, y_pred, x_sens,
                                           ('ppv', 'npv'),
                                           x_sens_privileged,
                                           labels,
                                           as_df)
