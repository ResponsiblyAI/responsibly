__all__ = ['build_FICO_dataset']


import numpy as np
import pandas as pd
from pkg_resources import resource_filename
from sklearn.metrics import auc


CDF_BY_RACE_PATH = resource_filename(__name__,
                                     'transrisk_cdf_by_race_ssa.csv')


PERFORMANCE_BY_RACE_PATH = resource_filename(__name__,
                                             'transrisk_performance_by_race_ssa.csv')  # pylint: disable=line-too-long

TOTAL_BY_RACE_PATH = resource_filename(__name__,
                                       'totals.csv')


def _cleanup_frame(frame):
    """Rename and re-order columns."""
    frame = frame.rename(columns={'Non- Hispanic white': 'White'})
    frame = frame.reindex(['Asian', 'Black', 'Hispanic', 'White'],
                          axis=1)
    return frame


def _read_totals():
    """Read the total number of people of each race."""
    frame = _cleanup_frame(pd.read_csv(TOTAL_BY_RACE_PATH, index_col=0))
    return {r: frame[r]['SSA'] for r in frame.columns}


def _parse_data():
    """Parse sqf data set."""
    cdfs = _cleanup_frame(pd.read_csv(CDF_BY_RACE_PATH, index_col=0))
    performance = (100
                   - _cleanup_frame(pd.read_csv(PERFORMANCE_BY_RACE_PATH,
                                                index_col=0)))
    return (cdfs / 100, performance / 100)


def _load_data():
    totals = _read_totals()
    cdfs_df, performance_df = _parse_data()
    return totals, cdfs_df, performance_df


def _get_pdfs(cdfs_df):
    cdf_vs = np.concatenate([[np.zeros_like(cdfs_df.values[0])],
                             cdfs_df.values])
    pdf_vs = (cdf_vs[1:] - cdf_vs[:-1])
    pdfs_df = pd.DataFrame(pdf_vs,
                           columns=cdfs_df.columns, index=cdfs_df.index)
    return pdfs_df


def _calc_tpr_fpr(pdfs_df, performance_df):
    dfs = []
    for value in [performance_df, 1 - performance_df]:
        proportion_per_score = value * pdfs_df

        proportion_over_all_scores = proportion_per_score.sum(axis=0)

        cum_prop_per_score = proportion_per_score[::-1].cumsum(axis=0)[::-1]

        rate = cum_prop_per_score / proportion_over_all_scores

        # by sklean convention, thresholds[0]
        # represents no instances being predicted positive
        # and is arbitrarily set to max(y_score) + 1
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        rate.loc[max(rate.index) + 1] = [0] * len(rate.columns)

        dfs.append(rate)

    tpr_df, fpr_df = dfs  # pylint: disable=unbalanced-tuple-unpacking
    return tpr_df, fpr_df


def _build_rocs(fpr_df, tpr_df):
    rocs = {}
    for group in fpr_df.columns:
        fprs = fpr_df[group].values[::-1]
        tprs = tpr_df[group].values[::-1]
        thresholds = fpr_df.index[::-1]

        rocs[group] = (fprs,
                       tprs,
                       thresholds)

    return rocs


def build_FICO_dataset():
    """Build the FICO dataset.

    Dataset of the credit score of TransUnion (called TransRisk).
    The TransRisk score is in turn based on
    a proprietary model created by FICO,
    hence often referred to as FICO scores.

    The data is *aggregated*, i.e., there is no outcome
    and prediction information per individual,
    but summarized statistics for each FICO score
    and race/race/ethnicity group.

    +---------------+------------------------------------------------------+
    | FICO key      | Meaning                                              |
    +===============+======================================================+
    | `total`       | Total number of individuals                          |
    +---------------+------------------------------------------------------+
    | `totals`      | Number of individuals per group                      |
    +---------------+------------------------------------------------------+
    | `cdf`         | Cumulative distribution function of score per group  |
    +---------------+------------------------------------------------------+
    | `pdf`         | Probability distribution function of score per group |
    +---------------+------------------------------------------------------+
    | `performance` | Fraction of non-defaulters per score and group       |
    +---------------+------------------------------------------------------+
    | `base_rates`  | Base rate of non-defaulters per group                |
    +---------------+------------------------------------------------------+
    | `base_rate`   | The overall base rate non-defaulters                 |
    +---------------+------------------------------------------------------+
    | `proportions` | Fraction of individuals per group                    |
    +---------------+------------------------------------------------------+
    | `fpr`         | True Positive Rate by score as threshold per group   |
    +---------------+------------------------------------------------------+
    | `tpr`         | False Positive Rate by score as threshold per group  |
    +---------------+------------------------------------------------------+
    | `rocs`        | ROC per group                                        |
    +---------------+------------------------------------------------------+
    | `aucs`        | ROC AUC per group                                    |
    +---------------+------------------------------------------------------+

    :return: Dictionary of various aggregated statics
             of the FICO credit score.
    :rtype: dict

    References:
        - Based on code (MIT License) by Moritz Hardt
          from https://github.com/fairmlbook/fairmlbook.github.io
        - https://fairmlbook.org/demographic.html#case-study-credit-scoring

    """

    totals, cdfs_df, performance_df = _load_data()
    pdfs_df = _get_pdfs(cdfs_df)

    total = sum(totals.values())

    proportions = {group: total / sum(totals.values())
                   for group, total in totals.items()}

    base_rates = (pdfs_df * performance_df).sum()
    base_rate = (base_rates * pd.Series(proportions)).sum()

    tpr_df, fpr_df = _calc_tpr_fpr(pdfs_df, performance_df)
    rocs = _build_rocs(fpr_df, tpr_df)

    aucs = {group: auc(fpr, tpr) for group, (fpr, tpr, _)
            in rocs.items()}

    return {'total': total,
            'totals': totals,
            'cdf': cdfs_df,
            'pdf': pdfs_df,
            'performance': performance_df,
            'base_rates': base_rates,
            'base_rate': base_rate,
            'proportions': proportions,
            'fpr': fpr_df,
            'tpr': tpr_df,
            'rocs': rocs,
            'aucs': aucs}
