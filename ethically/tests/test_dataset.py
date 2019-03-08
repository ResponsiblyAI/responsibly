"""Unit test module for ethically.datasets"""
# pylint: disable=redefined-outer-name

import pytest

from ethically.dataset import COMPASDataset, GermanDataset


@pytest.fixture
def compas_ds():
    return COMPASDataset()


@pytest.fixture
def german_ds():
    return GermanDataset()


def test_compas_validate(compas_ds):
    compas_ds._validate()   # pylint: disable=protected-access


def test_compas_str(compas_ds):
    assert (str(compas_ds)
            == '<ProPublica Recidivism/COMPAS.'
            ' 6172 rows, 55 columns'
            ' in which {race} are sensitive attributes>')


def test_german_validate(german_ds):
    german_ds._validate()   # pylint: disable=protected-access


def test_german_str(german_ds):
    assert (str(german_ds)
            == '<German Credit Dataset.'
            ' 1000 rows, 23 columns'
            ' in which {age_factor} are sensitive attributes>')
