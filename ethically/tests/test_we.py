"Unit test module for ethically.we.core "
# pylint: disable=redefined-outer-name,unused-variable,expression-not-assigned,singleton-comparison,protected-access

import os
from math import isclose

import numpy as np
import pytest
from gensim.models.keyedvectors import KeyedVectors
from pkg_resources import resource_filename

from ethically.we import GenderBiasWE
from ethically.we.utils import project_reject_vector, project_vector

from ..consts import RANDOM_STATE


ATOL = 1e-6
N_RANDOM_NEUTRAL_WORDS_DEBIAS_TO_TEST = 1000


@pytest.fixture
def gender_biased_we():
    # pylint: disable=C0301
    model = KeyedVectors.load_word2vec_format(
        resource_filename(__name__, os.path.join('data',
                                                 'GoogleNews-vectors-negative300-bolukbasi.bin')),
        binary=True)
    return GenderBiasWE(model)


def test_calc_direct_bias(gender_biased_we):
    """
    Test calc_direct_bias method in GenderBiasWE
    Based on section 5.2
    """

    # TODO: it seemse that in the article it was checked on
    # all the professions names including gender specific ones
    # (e.g. businesswomen)
    assert isclose(gender_biased_we.calc_direct_bias(), 0.07, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_direct_bias(gender_biased_we
                                                     .PROFESSIONS_NAME),
                   0.08, abs_tol=1e-2)


# TODO: iterate over a dictionary
def test_calc_indirect_bias(gender_biased_we, all_zero=False):
    """
    Test calc_direct_bias method in GenderBiasWE
    Based on figure 3 & section 3.5
    """
    assert isclose(gender_biased_we.calc_indirect_bias('softball',
                                                       'pitcher'),
                   0 if all_zero else -0.01, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('softball',
                                                       'bookkeeper'),
                   0 if all_zero else 0.20, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('softball',
                                                       'receptionist'),
                   0 if all_zero else 0.67, abs_tol=1e-2)
    # these words have legit gender direction projection
    if not all_zero:
        assert isclose(gender_biased_we.calc_indirect_bias('softball',
                                                           'registered_nurse'),
                       0 if all_zero else 0.29, abs_tol=1e-2)
        # TODO: in the article it is 0.35 - why?
        assert isclose(gender_biased_we.calc_indirect_bias('softball',
                                                           'waitress'),
                       0 if all_zero else 0.31, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('softball',
                                                       'homemaker'),
                   0 if all_zero else 0.38, abs_tol=1e-2)

    assert isclose(gender_biased_we.calc_indirect_bias('football',
                                                       'footballer'),
                   0 if all_zero else 0.02, abs_tol=1e-2)
    # this word have legit gender direction projection
    if not all_zero:
        # TODO in the article it is 0.31 - why?
        assert isclose(gender_biased_we.calc_indirect_bias('football',
                                                           'businessman'),
                       0 if all_zero else 0.17, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('football',
                                                       'pundit'),
                   0 if all_zero else 0.10, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('football',
                                                       'maestro'),
                   0 if all_zero else 0.41, abs_tol=1e-2)
    assert isclose(gender_biased_we.calc_indirect_bias('football',
                                                       'cleric'),
                   0 if all_zero else 0.02, abs_tol=1e-2)


def check_all_vectors_unit_length(bias_we):
    for word in bias_we.model.vocab:
        vector = bias_we[word]
        norm = (vector ** 2).sum()
        np.testing.assert_allclose(norm, 1, atol=ATOL)


def test_neutralize(gender_biased_we, is_preforming=True):
    """
    Test _neutralize method in GenderBiasWE
    """
    neutral_words = gender_biased_we.NEUTRAL_WORDS

    if is_preforming:
        gender_biased_we._neutralize(neutral_words)

    direction_projections = [project_vector(gender_biased_we[word],
                                            gender_biased_we.direction)
                             for word in neutral_words]

    np.testing.assert_allclose(direction_projections, 0, atol=ATOL)

    np.testing.assert_allclose(gender_biased_we.calc_direct_bias(), 0,
                               atol=ATOL)

    check_all_vectors_unit_length(gender_biased_we)
    test_calc_indirect_bias(gender_biased_we, all_zero=True)


def test_equalize(gender_biased_we, is_preforming=True):
    """
    Test _equalize method in GenderBiasWE
    """
    equality_sets = gender_biased_we.DEFINITIONAL_PAIRS

    if is_preforming:
        gender_biased_we._equalize(equality_sets)

    for equality_set in equality_sets:
        projection_vectors = []
        rejection_vectors = []

        for equality_word in equality_set:
            vector = gender_biased_we[equality_word]

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

    check_all_vectors_unit_length(gender_biased_we)

def test_hard_debias_inplace(gender_biased_we, is_preforming=True):
    """
    Test hard_debias method in GenderBiasWE
    """
    # pylint: disable=C0301
    if is_preforming:
        test_calc_direct_bias(gender_biased_we)
        gender_biased_we.debias(method='hard')

    test_neutralize(gender_biased_we, is_preforming=False)
    test_equalize(gender_biased_we, is_preforming=False)

    equality_sets = gender_biased_we.DEFINITIONAL_PAIRS

    np.random.seed(RANDOM_STATE)
    neutral_words = np.random.choice(gender_biased_we.NEUTRAL_WORDS,
                                     N_RANDOM_NEUTRAL_WORDS_DEBIAS_TO_TEST,
                                     replace=False)

    for neutral_word in neutral_words:
        for equality_word1, equality_word2 in equality_sets:

            we1 = gender_biased_we[neutral_word] @ gender_biased_we[equality_word1]
            we2 = gender_biased_we[neutral_word] @ gender_biased_we[equality_word2]
            np.testing.assert_allclose(we1, we2, atol=ATOL)

            we1_distance = np.linalg.norm(gender_biased_we[neutral_word]
                                          - gender_biased_we[equality_word1])
            we2_distance = np.linalg.norm(gender_biased_we[neutral_word]
                                          - gender_biased_we[equality_word2])

            np.testing.assert_allclose(we1_distance, we2_distance, atol=ATOL)


def test_hard_debias_not_inplace(gender_biased_we):
    test_calc_direct_bias(gender_biased_we)

    gender_debiased_we = gender_biased_we.debias(method='hard',
                                                 inplace=False)

    test_calc_direct_bias(gender_biased_we)
    test_hard_debias_inplace(gender_debiased_we, is_preforming=False)
