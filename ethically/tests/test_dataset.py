"""Unit test module for ethically.datasets"""
# pylint: disable=redefined-outer-name

import pytest

from ethically.dataset import COMPASDataset


@pytest.fixture
def compas_ds():
    return COMPASDataset()


def test_validate(compas_ds):
    compas_ds._validate()   # pylint: disable=protected-access
