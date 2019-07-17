from collections import defaultdict

import seaborn as sns
from matplotlib import pylab as plt

from responsibly.fairness.metrics.score import (
    roc_auc_score_by_attr, roc_curve_by_attr,
)


def _groupby(x, by):
    d = defaultdict(list)
    for key, val in zip(by, x):
        d[key].append(val)
    return d


def distplot_by(a, by, bins=None, hist=True, kde=True, rug=False,
                fit=None, hist_kws=None, kde_kws=None, rug_kws=None,
                fit_kws=None, vertical=False, norm_hist=False,
                ax=None):

    axes = [sns.distplot(a_group,
                         bins=bins, hist=hist, kde=kde, rug=rug,
                         fit=fit, hist_kws=hist_kws, kde_kws=kde_kws,
                         rug_kws=rug_kws, fit_kws=fit_kws,
                         vertical=vertical, norm_hist=norm_hist,
                         ax=ax, label=group)
            for group, a_group in _groupby(a, by).items()]
    plt.legend()
    return axes


# Soruce: https://github.com/reiinakano/scikit-plot/blob/master/scikitplot/metrics.py#L332
def plot_roc_curves(roc_curves, aucs=None,
                    title='ROC Curves by Attribute',
                    ax=None, figsize=None,
                    title_fontsize='large', text_fontsize='medium'):
    """Generate the ROC curves by attribute from (fpr, tpr, thresholds).

    Based on :func:`skplt.metrics.plot_roc`

    :param roc_curves: Receiver operating characteristic (ROC)
                       by attribute.
    :type roc_curves: dict
    :param aucs: Area Under the ROC (AUC) by attribute.
    :type aucs: dict
    :param str title: Title of the generated plot.
    :param ax: The axes upon which to plot the curve.
               If `None`, the plot is drawn on a new set of axes.
    :param tuple figsize: Tuple denoting figure size of the plot
                          e.g. (6, 6).
    :param title_fontsize: Matplotlib-style fontsizes.
                          Use e.g. 'small', 'medium', 'large'
                          or integer-values.
    :param text_fontsize: Matplotlib-style fontsizes.
                          Use e.g. 'small', 'medium', 'large'
                          or integer-values.
    :return: The axes on which the plot was drawn.
    :rtype: :class:`matplotlib.axes.Axes`

    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)  # pylint: disable=unused-variable

    ax.set_title(title, fontsize=title_fontsize)

    for x_sens_value in roc_curves:

        label = 'ROC curve of group {0}'.format(x_sens_value)
        if aucs is not None:
            label += ' (area = {:0.2f})'.format(aucs[x_sens_value])

        ax.plot(roc_curves[x_sens_value][0],
                roc_curves[x_sens_value][1],
                lw=2,
                label=label)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)

    return ax


def plot_roc_by_attr(y_true, y_score, x_sens,
                     title='ROC Curves by Attribute',
                     ax=None, figsize=None,
                     title_fontsize='large', text_fontsize='medium'):
    """Generate the ROC curves by attribute from targets and scores.

    Based on :func:`skplt.metrics.plot_roc`

    :param y_true: Binary ground truth (correct) target values.
    :param y_score: Estimated target score as returned by a classifier.
    :param x_sens: Sensitive attribute values corresponded to each
                   estimated target.
    :param str title: Title of the generated plot.
    :param ax: The axes upon which to plot the curve.
               If `None`, the plot is drawn on a new set of axes.
    :param tuple figsize: Tuple denoting figure size of the plot
                          e.g. (6, 6).
    :param title_fontsize: Matplotlib-style fontsizes.
                          Use e.g. 'small', 'medium', 'large'
                          or integer-values.
    :param text_fontsize: Matplotlib-style fontsizes.
                          Use e.g. 'small', 'medium', 'large'
                          or integer-values.
    :return: The axes on which the plot was drawn.
    :rtype: :class:`matplotlib.axes.Axes`

    """

    roc_curves = roc_curve_by_attr(y_true, y_score, x_sens)
    aucs = roc_auc_score_by_attr(y_true, y_score, x_sens)
    return plot_roc_curves(roc_curves, aucs,
                           title, ax, figsize,
                           title_fontsize, text_fontsize)
