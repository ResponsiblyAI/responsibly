"""Unit test module for responsibly.datasets"""
# pylint: disable=redefined-outer-name

import pytest

from responsibly.dataset import AdultDataset, COMPASDataset, GermanDataset


@pytest.fixture
def compas_ds():
    return COMPASDataset()


@pytest.fixture
def german_ds():
    return GermanDataset()


@pytest.fixture
def adult_ds():
    return AdultDataset()


def test_compas_validate(compas_ds):
    compas_ds._validate()   # pylint: disable=protected-access


def test_compas_str(compas_ds):
    assert (str(compas_ds)
            == '<ProPublica Recidivism/COMPAS Dataset.'
            ' 6172 rows, 56 columns'
            ' in which {race, sex} are sensitive attributes>')


def test_german_validate(german_ds):
    german_ds._validate()   # pylint: disable=protected-access


def test_german_str(german_ds):
    assert (str(german_ds)
            == '<German Credit Dataset.'
            ' 1000 rows, 23 columns'
            ' in which {age_factor} are sensitive attributes>')


def test_adult_validate(adult_ds):
    adult_ds._validate()   # pylint: disable=protected-access


def test_adult_str(adult_ds):
    assert (str(adult_ds)
            == '<Adult Dataset.'
            ' 45222 rows, 15 columns'
            ' in which {sex, race} are sensitive attributes>')
