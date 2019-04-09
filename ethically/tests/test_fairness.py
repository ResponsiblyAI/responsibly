"""Unit test module for ethically.fairness"""
# pylint: disable=redefined-outer-name,line-too-long

import pytest

from ethically.dataset import COMPASDataset, build_FICO_dataset
from ethically.fairness.interventions import threshold
from ethically.fairness.metrics import (
    independence_binary, plot_roc_by_attr, separation_binary,
    sufficiency_binary,
)
from ethically.fairness.metrics.binary import compare_privileged
from ethically.tests.utils import assert_deep_almost_equal


COST_MATRIX = [[0, - 5 / 6], [0, 1 / 6]]

FICO_SINGLE = (42.5,
               {'Asian': (0.1565417064845852, 0.7935353629697238),
                'Black': (0.03421435054752242, 0.5140401023192323),
                'Hispanic': (0.07888468140621363, 0.6254574967221331),
                'White': (0.10256570907095985, 0.7931580812742784)},
               -0.06830064096108142)

FICO_MIN_COST = ({'Asian': 38.5,
                  'Black': 49.0,
                  'Hispanic': 46.5,
                  'White': 41.0},
                 {'Asian': (0.18460486111161467, 0.8381280415042232),
                  'Black': (0.020067107153418196, 0.4079222735623631),
                  'Hispanic': (0.058653539271323774, 0.5589875936027981),
                  'White': (0.11087988887409528, 0.8075167832686126)},
                 -0.20640680666666658)

FICO_INDEPENDENCE = ({'Asian': 53.0,
                      'Black': 18.0,
                      'Hispanic': 31.5,
                      'White': 52.0},
                     {'Asian': (0.08776430097924892, 0.6420627855534078),
                      'Black': (0.33376894053666994, 0.9049491843143592),
                      'Hispanic': (0.1693120676295986, 0.7999748148855363),
                      'White': (0.058478838627887275, 0.6773498273098238)},
                     0.5189755206777411)

FICO_FNR = ({'Asian': 46.0, 'Black': 29.5, 'Hispanic': 35.0, 'White': 46.5},
            {'Asian': (0.13087886175132435, 0.7450217322938637),
             'Black': (0.1077846495570669, 0.7487442676155007),
             'Hispanic': (0.13118970292760168, 0.7416493778636012),
             'White': (0.07995200854013917, 0.7456505446682594)},
            -0.06352608487155771,
            0.7413488681028431)

FICO_SEPARATION = ({},
                   {'': (0.11393540338532637, 0.7091695018283077)},
                   0.054347341376646846)

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
    assert_deep_almost_equal(threshold.find_single_threshold(fico['rocs'],
                                                             fico['base_rates'],
                                                             fico['proportions'],
                                                             COST_MATRIX),
                             FICO_SINGLE)


# https://github.com/fairmlbook/fairmlbook.github.io/blob/b1dfc9756ee8d2adf502c1b0f7265f10c5b1033c/code/creditscore/criteria.py#L122
# in the github repo of fairmlbook, for max profit case
# the threshold for Hispanic is `47.0`,
# and not `46.5` as we get with our method.
# maybe it is because we are using trinary search for finding the minimum
# on the cost function, and in the fairmlbook there are performing argmin
# all over the cost function values.
def test_min_cost_threshold(fico):
    assert_deep_almost_equal(threshold.find_min_cost_thresholds(fico['rocs'],
                                                                fico['base_rates'],
                                                                COST_MATRIX),
                             FICO_MIN_COST)


def test_independence_thresholds(fico):
    assert_deep_almost_equal(threshold.find_independence_thresholds(fico['rocs'],
                                                                    fico['base_rates'],
                                                                    fico['proportions'],
                                                                    COST_MATRIX),
                             FICO_INDEPENDENCE)


def test_fnr_thresholds(fico):
    assert_deep_almost_equal(threshold.find_fnr_thresholds(fico['rocs'],
                                                           fico['base_rates'],
                                                           fico['proportions'],
                                                           COST_MATRIX),
                             FICO_FNR)


def test_separation_thresholds(fico):
    assert_deep_almost_equal(threshold.find_separation_thresholds(fico['rocs'],
                                                                  fico['base_rate'],
                                                                  COST_MATRIX),
                             FICO_SEPARATION)


def test_thresholds(fico):
    threshold_data = threshold.find_thresholds(fico['rocs'],
                                               fico['proportions'],
                                               fico['base_rate'],
                                               fico['base_rates'],
                                               COST_MATRIX)

    assert_deep_almost_equal(threshold_data, FICO_THRESHOLD_DATA)


def test_plot_roc_by_attr_thresholds_exception(compas_ds):
    df_missing_score = compas_ds.df[(((compas_ds.df['race'] == 'Caucasian')
                                      & (compas_ds.df['decile_score'] != 5))
                                     | (compas_ds.df['race'] == 'African-American'))]

    with pytest.raises(NotImplementedError):
        plot_roc_by_attr(df_missing_score['two_year_recid'],
                         df_missing_score['decile_score'],
                         df_missing_score['race'])
