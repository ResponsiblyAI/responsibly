"Unit test module for ethically.we.tolga "
# pylint: disable=redefined-outer-name,unused-variable,expression-not-assigned,singleton-comparison

import os
from math import isclose

import pytest
from gensim.models.keyedvectors import KeyedVectors
from pkg_resources import resource_filename

from ethically.we import GenderBiasWE


@pytest.fixture
def gender_biased_we():
    model = KeyedVectors.load_word2vec_format(
        resource_filename(__name__, os.path.join('data',
                'GoogleNews-vectors-negative300-tolga.bin')),
         binary=True)
    return GenderBiasWE(model)


def test_calc_direct_bias(gender_biased_we):
    """
    Test calc_direct_bias method in GenderBiasWE
    Based on section 5.2
    """
    assert isclose(gender_biased_we.calc_direct_bias(), 0.08, abs_tol=1e-2)


def test_calc_indirect_bias(gender_biased_we):
    """
    Test calc_direct_bias method in GenderBiasWE
    Based on figure 3 & section 3.5
    """
    assert isclose(gender_biased_we.calc_indirect_bias('softball', 'pitcher'),
                    -0.01, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('softball', 'bookkeeper'),
                    0.20, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('softball', 'receptionist'),
                    0.67, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('softball', 'registered_nurse'),
                    0.29, abs_tol=1e-2)
    # TODO: in the article it is 0.35 - why?
    assert isclose(gender_biased_we.calc_indirect_bias('softball', 'waitress'),
                    0.31, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('softball', 'homemaker'),
                    0.38, abs_tol=1e-2)

    assert isclose(gender_biased_we.calc_indirect_bias('football', 'footballer'),
                    0.02, abs_tol=1e-2)
    # TODO in the article it is 0.31 - why?
    assert isclose(gender_biased_we.calc_indirect_bias('football', 'businessman'),
                    0.17, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('football', 'pundit'),
                    0.10, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('football', 'maestro'),
                    0.41, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('football', 'cleric'),
                    0.02, abs_tol=1e-2)
