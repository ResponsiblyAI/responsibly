"""Unit test module for responsibly.fairness

FICO results are based on running the code from:
https://github.com/fairmlbook/fairmlbook.github.io
"""
# pylint: disable=redefined-outer-name,line-too-long

import pytest

from responsibly.dataset import COMPASDataset, build_FICO_dataset
from responsibly.fairness.interventions import threshold
from responsibly.fairness.metrics import (
    independence_binary, plot_roc_by_attr, separation_binary,
    sufficiency_binary,
)
from responsibly.fairness.metrics.binary import compare_privileged
from responsibly.tests.utils import assert_deep_almost_equal


FICO_TOL = {'atol': 1e-5, 'rtol': 5e-3}

COST_MATRIX = [[0, - 5 / 6], [0, 1 / 6]]

FICO_SINGLE = (42.5,
               {'Asian': (0.159210978241477, 0.7982257394726964),
                'Black': (0.03562475202516957, 0.5222536366818645),
                'Hispanic': (0.0811160539719404, 0.6318581559145164),
                'White': (0.10494970149747886, 0.7974084986440749)},
               -11888.109525713331 / 174047)

# In the FairMLBook repo the cutoffs are
# {'Asian': 38.5, 'Black': 49.0, 'Hispanic': 47.0, 'White': 41.0}
# Cost per group from there:
# max_profit Asian 0.0027352900000000044
# max_profit Black 1.0889999999999889e-05
# max_profit Hispanic 9.025000000000055e-05
# max_profit White 0.00021315999999999818
# Cost is NOT taken from FairMLBook repo
FICO_MIN_COST = ({'Asian': 38.5,
                  'Black': 49.0,
                  'Hispanic': 47.0,
                  'White': 41.0},
                 {'Asian': (0.18688340039863297, 0.8409289340716826),
                  'Black': (0.020748274408693028, 0.4148991737082274),
                  'Hispanic': (0.05865353927132363, 0.5589875936027977),
                  'White': (0.11484787793720092, 0.8138995270074796)},
                 -0.06876633661556937)


# In the FairMLBook repo the cutoffs are
# {'Asian': 53.5, 'Black': 18.5, 'Hispanic': 32.0, 'White': 52.5}
# and the acceptance rate is 0.5189755206777411
# It might be that there is some issue with _ternary_search_float
# The cost is slightly worse, so we probably don't find the optimal
# solution, but very close to it
# From the repo this is the cost: -8183.343726279998 / 174047
FICO_INDEPENDENCE = ({'Asian': 53.5,
                      'Black': 18.0,
                      'Hispanic': 31.5,
                      'White': 52.0},
                     {'Asian': (0.08776430097924894, 0.6420627855534083),
                      'Black': (0.3445144172941361, 0.908131231733616),
                      'Hispanic': (0.1739271675548935, 0.8061475266133501),
                      'White': (0.060298201269177995, 0.6844160268999959)},
                     -0.046725728876261405,
                     0.532796101034295)

# All the values are NOT taken from FairMLBook repo
FICO_FNR = ({'Asian': 45.5, 'Black': 29.5, 'Hispanic': 35.0, 'White': 46},
            {'Asian': (0.13830821486672065, 0.7597270937393894),
             'Black': (0.11069747158860148, 0.753996048320934),
             'Hispanic': (0.13779115742038225, 0.7528248235528201),
             'White': (0.0845338139109347, 0.7565831625671293)},
            -0.06352608487155771,
            0.2402515268107051)

FICO_SEPARATION = ({},
                   {'': (0.11130392607695799, 0.7032306899595236)},
                   # slightly different from the fairmlbook repository:
                   -9448.09030585854 / 174047)

FICO_THRESHOLD_DATA = {'single': FICO_SINGLE,
                       'min_cost': FICO_MIN_COST,
                       'independence': FICO_INDEPENDENCE,
                       'fnr': FICO_FNR,
                       'separation': FICO_SEPARATION}


@pytest.fixture
def compas_ds():
    ds = COMPASDataset()
    df = ds.df
    df = df[df['race'].isin(['African-American',
                             'Caucasian'])]
    ds.df = df
    return ds


def test_independence_binary(compas_ds):
    indp, _ = independence_binary((compas_ds.df['y_pred']),
                                  compas_ds.df['race'])
    assert (indp['African-American']['acceptance_rate']
            == pytest.approx(0.576, abs=0.001))
    assert (indp['Caucasian']['acceptance_rate']
            == pytest.approx(0.33, abs=0.001))


def test_sufficiency_binary(compas_ds):
    suf, _ = sufficiency_binary(compas_ds.df['two_year_recid'],
                                compas_ds.df['y_pred'],
                                compas_ds.df['race'])
    assert (suf['African-American']['ppv']
            == pytest.approx(0.649, abs=0.001))
    assert (suf['Caucasian']['ppv']
            == pytest.approx(0.594, abs=0.001))
    assert (suf['African-American']['npv']
            == pytest.approx(0.648, abs=0.001))
    assert (suf['Caucasian']['npv']
            == pytest.approx(0.710, abs=0.001))


def test_separation_binary(compas_ds):
    sep, _ = separation_binary(compas_ds.df['two_year_recid'],
                               compas_ds.df['y_pred'],
                               compas_ds.df['race'])
    assert (sep['African-American']['tpr']
            == pytest.approx(0.715, abs=0.001))
    assert (sep['Caucasian']['tpr']
            == pytest.approx(0.503, abs=0.001))
    assert (sep['African-American']['fpr']
            == pytest.approx(0.423, abs=0.001))
    assert (sep['Caucasian']['fpr']
            == pytest.approx(0.220, abs=0.001))


def test_compare_privileged():
    assert (compare_privileged({'A': {'x': 1, 'y': 40},
                                'B': {'x': 4, 'y': 5}},
                               'B')
            == {'metrics': {'x': {'diff': -3, 'ratio': 0.25},
                            'y': {'diff': 35, 'ratio': 8.0}},
                'x_sens_privileged': 'B',
                'x_sens_unprivileged': 'A'})


@pytest.fixture
def fico():
    return build_FICO_dataset()


def test_single_threshold(fico):
    assert_deep_almost_equal(FICO_SINGLE,
                             threshold.find_single_threshold(fico['rocs'],
                                                             fico['base_rates'],
                                                             fico['proportions'],
                                                             COST_MATRIX),
                             **FICO_TOL)


# https://github.com/fairmlbook/fairmlbook.github.io/blob/b1dfc9756ee8d2adf502c1b0f7265f10c5b1033c/code/creditscore/criteria.py#L122
# in the github repo of fairmlbook, for max profit case
# the threshold for Hispanic is `47.0`,
# and not `46.5` as we get with our method.
# maybe it is because we are using trinary search for finding the minimum
# on the cost function, and in the fairmlbook there are performing argmin
# all over the cost function values.
def test_min_cost_threshold(fico):
    assert_deep_almost_equal(FICO_MIN_COST,
                             threshold.find_min_cost_thresholds(fico['rocs'],
                                                                fico['base_rates'],
                                                                fico['proportions'],
                                                                COST_MATRIX),
                             **FICO_TOL)


def test_independence_thresholds(fico):
    assert_deep_almost_equal(FICO_INDEPENDENCE,
                             threshold.find_independence_thresholds(fico['rocs'],
                                                                    fico['base_rates'],
                                                                    fico['proportions'],
                                                                    COST_MATRIX),
                             **FICO_TOL)


def test_fnr_thresholds(fico):
    assert_deep_almost_equal(FICO_FNR,
                             threshold.find_fnr_thresholds(fico['rocs'],
                                                           fico['base_rates'],
                                                           fico['proportions'],
                                                           COST_MATRIX),
                             **FICO_TOL)


def test_separation_thresholds(fico):
    assert_deep_almost_equal(FICO_SEPARATION,
                             threshold.find_separation_thresholds(fico['rocs'],
                                                                  fico['base_rate'],
                                                                  COST_MATRIX),
                             **FICO_TOL)


def test_thresholds(fico):
    threshold_data = threshold.find_thresholds(fico['rocs'],
                                               fico['proportions'],
                                               fico['base_rate'],
                                               fico['base_rates'],
                                               COST_MATRIX)

    assert_deep_almost_equal(FICO_THRESHOLD_DATA, threshold_data,
                             **FICO_TOL)


def test_plot_roc_by_attr_thresholds_exception(compas_ds):
    df_missing_score = compas_ds.df[(((compas_ds.df['race'] == 'Caucasian')
                                      & (compas_ds.df['decile_score'] != 5))
                                     | (compas_ds.df['race'] == 'African-American'))]

    with pytest.raises(NotImplementedError):
        plot_roc_by_attr(df_missing_score['two_year_recid'],
                         df_missing_score['decile_score'],
                         df_missing_score['race'])
