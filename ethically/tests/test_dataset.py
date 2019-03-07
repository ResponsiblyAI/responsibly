"""Unit test module for ethically.datasets"""
# pylint: disable=redefined-outer-name

import pytest

from ethically.dataset import COMPASDataset


@pytest.fixture
def compas_ds():
    return COMPASDataset()


def test_compas_validate(compas_ds):
    compas_ds._validate()   # pylint: disable=protected-access


def test_compas_str(compas_ds):
    assert (str(compas_ds)
            == '<ProPublica Recidivism/COMPAS.'
            ' 6172 rows, 55 columns'
            ' in which {race} are sensitive attributes>')
