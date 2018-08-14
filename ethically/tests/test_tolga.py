"Unit test module for ethically.we.tolga "
# pylint: disable=redefined-outer-name,unused-variable,expression-not-assigned,singleton-comparison

import os
from math import isclose

import numpy as np
import pytest
from gensim.models.keyedvectors import KeyedVectors
from pkg_resources import resource_filename

from ethically.we import GenderBiasWE
from ethically.we.utils import project_reject_vector, project_vector


ATOL = 1e-6


@pytest.fixture
def gender_biased_we():
    # pylint: disable=C0301
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

    # it seemse that in the article it was checked on all the professions names
    # including gender specific ones (e.g. businesswomen)
    assert isclose(gender_biased_we.calc_direct_bias(), 0.07, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_direct_bias(gender_biased_we
                                                     .PROFESSIONS_NAME),
                   0.08, abs_tol=1e-2)


# TODO: iterate over a dictionary
def test_calc_indirect_bias(gender_biased_we):
    """
    Test calc_direct_bias method in GenderBiasWE
    Based on figure 3 & section 3.5
    """
    assert isclose(gender_biased_we.calc_indirect_bias('softball',
                                                       'pitcher'),
                   -0.01, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('softball',
                                                       'bookkeeper'),
                   0.20, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('softball',
                                                       'receptionist'),
                   0.67, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('softball',
                                                       'registered_nurse'),
                   0.29, abs_tol=1e-2)
    # TODO: in the article it is 0.35 - why?
    assert isclose(gender_biased_we.calc_indirect_bias('softball',
                                                       'waitress'),
                   0.31, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('softball',
                                                       'homemaker'),
                   0.38, abs_tol=1e-2)

    assert isclose(gender_biased_we.calc_indirect_bias('football',
                                                       'footballer'),
                   0.02, abs_tol=1e-2)
    # TODO in the article it is 0.31 - why?
    assert isclose(gender_biased_we.calc_indirect_bias('football',
                                                       'businessman'),
                   0.17, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('football',
                                                       'pundit'),
                   0.10, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('football',
                                                       'maestro'),
                   0.41, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('football',
                                                       'cleric'),
                   0.02, abs_tol=1e-2)


def test_neutralize(gender_biased_we):
    """
    Test _neutralize method in GenderBiasWE
    """
    neutral_words = gender_biased_we.PROFESSIONS_NAME
    gender_biased_we._neutralize(neutral_words)  # pylint: disable=W0212
    direction_projections = [project_vector(gender_biased_we.model[word],
                                            gender_biased_we.direction)
                             for word in neutral_words]
    np.testing.assert_allclose(direction_projections, 0, atol=ATOL)

    np.testing.assert_allclose(gender_biased_we.calc_direct_bias(), 0,
                               atol=ATOL)


def test_equalize(gender_biased_we):
    equality_sets = gender_biased_we.DEFINITIONAL_PAIRS
    gender_biased_we._equalize(equality_sets)  # pylint: disable=W0212

    for equality_set in equality_sets:
        projection_vectors = []
        rejection_vectors = []

        for equality_word in equality_set:
            vector = gender_biased_we.model[equality_word]

            np.testing.assert_allclose(np.linalg.norm(vector), 1, atol=ATOL)

            # pylint: disable=C0301
            (projection_vector,
             rejection_vector) = project_reject_vector(vector,
                                                       gender_biased_we.direction)
            projection_vectors.append(projection_vector)
            rejection_vectors.append(rejection_vector)

        for rejection_vector in rejection_vectors[1:]:
            np.testing.assert_allclose(rejection_vectors[0],
                                       rejection_vector,
                                       atol=ATOL)


def test_hard_debias(gender_biased_we):
    # pylint: disable=C0301

    gender_biased_we.debias(method='hard')
    test_neutralize(gender_biased_we)
    # test_equalize(gender_biased_we)

    equality_sets = gender_biased_we.DEFINITIONAL_PAIRS
    neutral_words = gender_biased_we.PROFESSIONS_NAME

    for neutral_word in neutral_words:
        for equality_word1, equality_word2 in equality_sets:

            we1 = gender_biased_we.model[neutral_word] @ gender_biased_we.model[equality_word1]
            we2 = gender_biased_we.model[neutral_word] @ gender_biased_we.model[equality_word2]
            np.testing.assert_allclose(we1, we2, atol=ATOL)

            we1_distance = np.linalg.norm(gender_biased_we.model[neutral_word]
                                          - gender_biased_we.model[equality_word1])
            we2_distance = np.linalg.norm(gender_biased_we.model[neutral_word]
                                          - gender_biased_we.model[equality_word2])

            np.testing.assert_allclose(we1_distance, we2_distance, atol=ATOL)
