"""Unit test module for responsibly.we"""
# pylint: disable=redefined-outer-name,unused-variable,expression-not-assigned,singleton-comparison,protected-access

import copy
from math import isclose

import numpy as np
import pytest

from responsibly.consts import RANDOM_STATE
from responsibly.tests.data import TOLGA_GENDER_ANALOGIES
from responsibly.tests.utils import assert_deep_almost_equal
from responsibly.we import (
    GenderBiasWE, calc_all_weat, calc_weat_pleasant_unpleasant_attribute,
)
from responsibly.we.data import WEAT_DATA, load_w2v_small
from responsibly.we.utils import (
    most_similar, normalize, project_params, project_reject_vector,
    project_vector,
)


ATOL = 1e-6

N_RANDOM_NEUTRAL_WORDS_DEBIAS_TO_TEST = 1000


@pytest.fixture
def w2v_small():
    return load_w2v_small()


@pytest.fixture
def gender_biased_w2v_small():
    model = load_w2v_small()
    return GenderBiasWE(model, only_lower=True, verbose=True)


def test_identify_direction_single(w2v_small):
    gb = GenderBiasWE(w2v_small, only_lower=True, verbose=True,
                      identify_direction='single')

    direction = normalize(w2v_small['she'] - w2v_small['he'])

    np.testing.assert_allclose(gb.direction, direction)


# TODO: only check that there is no exception,
# should b change to a better test cas
def test_identify_direction_sum(w2v_small):
    gb = GenderBiasWE(w2v_small, only_lower=True, verbose=True,
                      identify_direction='sum')


def test_assert_gensim_keyed_vectors():
    with pytest.raises(TypeError):
        GenderBiasWE(['one', 'two'], only_lower=True, verbose=True)


def test_project_params():
    # pylint: disable=arguments-out-of-order
    v = np.array([1, 2, 3])
    u = np.array([-4, 5, -6])

    (_,
     projected_vector_v1,
     rejected_vector_v1) = project_params(v, u)

    projected_vector_v2, rejected_vector_v2 = project_reject_vector(u, v)

    np.testing.assert_allclose(projected_vector_v1, projected_vector_v2)
    np.testing.assert_allclose(rejected_vector_v1, rejected_vector_v2)


def test_words_embbeding_loading(gender_biased_w2v_small):
    assert len(gender_biased_w2v_small.model.vocab) == 26423


def test_contains(gender_biased_w2v_small):
    assert 'home' in gender_biased_w2v_small
    assert 'HOME' not in gender_biased_w2v_small


def test_data_is_sorted_list(gender_biased_w2v_small):
    # otherwise 'specific_full_with_definitional_equalize' is not sorted
    assert gender_biased_w2v_small.only_lower

    for key in gender_biased_w2v_small._data['word_group_keys']:
        word_list = gender_biased_w2v_small._data[key]
        assert isinstance(word_list, list)
        assert all(word_list[i] <= word_list[i + 1]
                   for i in range(len(word_list) - 1))


def test_calc_direct_bias(gender_biased_w2v_small):
    """
    Test calc_direct_bias method in GenderBiasWE.

    Based on section 5.2
    """

    # TODO: it seemse that in the article it was checked on
    # all the professions names including gender specific ones
    # (e.g. businesswomen)
    assert isclose(gender_biased_w2v_small.calc_direct_bias(),
                   0.07, abs_tol=1e-2)
    assert isclose(gender_biased_w2v_small.calc_direct_bias(gender_biased_w2v_small  # pylint: disable=C0301
                                                            ._data['profession_names']),  # pylint: disable=C0301

                   0.08, abs_tol=1e-2)


# TODO: iterate over a dictionary
def test_calc_indirect_bias(gender_biased_w2v_small, all_zero=False):
    """
    Test calc_direct_bias method in GenderBiasWE.

    Based on figure 3 & section 3.5
    """
    assert isclose(gender_biased_w2v_small.calc_indirect_bias('softball',
                                                              'pitcher'),
                   0 if all_zero else -0.01, abs_tol=1e-2)
    assert isclose(gender_biased_w2v_small.calc_indirect_bias('softball',
                                                              'bookkeeper'),
                   0 if all_zero else 0.20, abs_tol=1e-2)
    assert isclose(gender_biased_w2v_small.calc_indirect_bias('softball',
                                                              'receptionist'),
                   0 if all_zero else 0.67, abs_tol=1e-2)
    # these words have legit gender direction projection
    if not all_zero:
        assert isclose(gender_biased_w2v_small.calc_indirect_bias('softball',
                                                                  'registered_nurse'),  # pylint: disable=C0301
                       0 if all_zero else 0.29, abs_tol=1e-2)
        # TODO: in the article it is 0.35 - why?
        assert isclose(gender_biased_w2v_small.calc_indirect_bias('softball',
                                                                  'waitress'),
                       0 if all_zero else 0.31, abs_tol=1e-2)
    assert isclose(gender_biased_w2v_small.calc_indirect_bias('softball',
                                                              'homemaker'),
                   0 if all_zero else 0.38, abs_tol=1e-2)

    assert isclose(gender_biased_w2v_small.calc_indirect_bias('football',
                                                              'footballer'),
                   0 if all_zero else 0.02, abs_tol=1e-2)
    # this word have legit gender direction projection
    if not all_zero:
        # TODO in the article it is 0.31 - why?
        assert isclose(gender_biased_w2v_small.calc_indirect_bias('football',
                                                                  'businessman'),  # pylint: disable=C0301
                       0 if all_zero else 0.17, abs_tol=1e-2)
    assert isclose(gender_biased_w2v_small.calc_indirect_bias('football',
                                                              'pundit'),
                   0 if all_zero else 0.10, abs_tol=1e-2)
    assert isclose(gender_biased_w2v_small.calc_indirect_bias('football',
                                                              'maestro'),
                   0 if all_zero else 0.41, abs_tol=1e-2)
    assert isclose(gender_biased_w2v_small.calc_indirect_bias('football',
                                                              'cleric'),
                   0 if all_zero else 0.02, abs_tol=1e-2)


def test_generate_closest_words_indirect_bias(gender_biased_w2v_small):
    """Test generate_closest_words_indirect_bias in GenderBiasWE."""
    result = {'indirect_bias': {('football', 'businessman'): 0.17,
                                ('football', 'cleric'): 0.02,
                                ('football', 'footballer'): 0.02,
                                ('football', 'maestro'): 0.42,
                                ('football', 'pundit'): 0.1,
                                ('softball', 'bookkeeper'): 0.2,
                                ('softball', 'paralegal'): 0.37,
                                ('softball', 'receptionist'): 0.67,
                                ('softball', 'registered_nurse'): 0.29,
                                ('softball', 'waitress'): 0.32},
              'projection': {('football', 'businessman'): -0.2,
                             ('football', 'cleric'): -0.17,
                             ('football', 'footballer'): -0.34,
                             ('football', 'maestro'): -0.18,
                             ('football', 'pundit'): -0.19,
                             ('softball', 'bookkeeper'): 0.18,
                             ('softball', 'paralegal'): 0.14,
                             ('softball', 'receptionist'): 0.16,
                             ('softball', 'registered_nurse'): 0.16,
                             ('softball', 'waitress'): 0.15}}

    indirect_bias_df = (gender_biased_w2v_small
                        .generate_closest_words_indirect_bias('softball',
                                                              'football'))
    assert (indirect_bias_df
            .round(2)
            .to_dict()) == result


def check_all_vectors_unit_length(bias_we):
    for word in bias_we.model.vocab:
        vector = bias_we[word]
        norm = (vector ** 2).sum()
        np.testing.assert_allclose(norm, 1, atol=ATOL)


def test_neutralize(gender_biased_w2v_small, is_preforming=True):
    """Test _neutralize method in GenderBiasWE."""
    neutral_words = gender_biased_w2v_small._data['neutral_words']

    if is_preforming:
        gender_biased_w2v_small._neutralize(neutral_words)

    direction_projections = [project_vector(gender_biased_w2v_small[word],
                                            gender_biased_w2v_small.direction)
                             for word in neutral_words]

    np.testing.assert_allclose(direction_projections, 0, atol=ATOL)

    np.testing.assert_allclose(gender_biased_w2v_small.calc_direct_bias(), 0,
                               atol=ATOL)

    check_all_vectors_unit_length(gender_biased_w2v_small)
    test_calc_indirect_bias(gender_biased_w2v_small, all_zero=True)


def test_equalize(gender_biased_w2v_small, is_preforming=True):
    """Test _equalize method in GenderBiasWE."""
    # pylint: disable=line-too-long
    equality_sets = {tuple(w) for w in gender_biased_w2v_small._data['equalize_pairs']}
    equality_sets |= {tuple(w) for w in gender_biased_w2v_small._data['definitional_pairs']}
    equality_sets = gender_biased_w2v_small._generate_pair_candidates(equality_sets)

    if is_preforming:
        gender_biased_w2v_small._equalize(equality_sets)

    for equality_set in equality_sets:
        projection_vectors = []
        rejection_vectors = []

        for equality_word in equality_set:
            vector = gender_biased_w2v_small[equality_word]

            np.testing.assert_allclose(np.linalg.norm(vector), 1, atol=ATOL)

            (projection_vector,
             rejection_vector) = project_reject_vector(vector,
                                                       gender_biased_w2v_small.direction)
            projection_vectors.append(projection_vector)
            rejection_vectors.append(rejection_vector)

        # <e1, d> == -<e2, d>
        # assuming equality sets of size 2
        assert len(projection_vectors) == 2
        np.testing.assert_allclose(projection_vectors[0] @ gender_biased_w2v_small.direction,
                                   -projection_vectors[1] @ gender_biased_w2v_small.direction,
                                   atol=ATOL)

        # all rejection part is equal for all the vectors
        for rejection_vector in rejection_vectors[1:]:
            np.testing.assert_allclose(rejection_vectors[0],
                                       rejection_vector,
                                       atol=ATOL)

    check_all_vectors_unit_length(gender_biased_w2v_small)


def test_hard_debias_inplace(gender_biased_w2v_small, is_preforming=True):
    """Test hard_debias method in GenderBiasWE."""
    # pylint: disable=C0301
    if is_preforming:
        test_calc_direct_bias(gender_biased_w2v_small)
        gender_biased_w2v_small.debias(method='hard')

    test_neutralize(gender_biased_w2v_small, is_preforming=False)
    test_equalize(gender_biased_w2v_small, is_preforming=False)

    equality_sets = gender_biased_w2v_small._data['definitional_pairs']

    np.random.seed(RANDOM_STATE)
    neutral_words = np.random.choice(gender_biased_w2v_small._data['neutral_words'],
                                     N_RANDOM_NEUTRAL_WORDS_DEBIAS_TO_TEST,
                                     replace=False)

    # for every neutal word w: <e1, w> == <e2, w> AND ||e1 - w|| == ||e2 - w||
    for neutral_word in neutral_words:
        for equality_word1, equality_word2 in equality_sets:

            we1 = gender_biased_w2v_small[neutral_word] @ gender_biased_w2v_small[equality_word1]
            we2 = gender_biased_w2v_small[neutral_word] @ gender_biased_w2v_small[equality_word2]
            np.testing.assert_allclose(we1, we2, atol=ATOL)

            we1_distance = np.linalg.norm(gender_biased_w2v_small[neutral_word]
                                          - gender_biased_w2v_small[equality_word1])
            we2_distance = np.linalg.norm(gender_biased_w2v_small[neutral_word]
                                          - gender_biased_w2v_small[equality_word2])

            np.testing.assert_allclose(we1_distance, we2_distance, atol=ATOL)


def test_hard_debias_not_inplace(gender_biased_w2v_small):
    test_calc_direct_bias(gender_biased_w2v_small)

    gender_debiased_we = gender_biased_w2v_small.debias(method='hard',
                                                        inplace=False)

    test_calc_direct_bias(gender_biased_w2v_small)
    test_hard_debias_inplace(gender_debiased_we, is_preforming=False)


def test_copy(gender_biased_w2v_small):
    gender_biased_w2v_small_copy = copy.copy(gender_biased_w2v_small)
    assert (gender_biased_w2v_small.direction
            is not gender_biased_w2v_small_copy.direction)
    assert gender_biased_w2v_small.model is gender_biased_w2v_small_copy.model


def test_deepcopy(gender_biased_w2v_small):
    gender_biased_w2v_small_copy = copy.deepcopy(gender_biased_w2v_small)
    assert (gender_biased_w2v_small.direction
            is not gender_biased_w2v_small_copy.direction)
    assert (gender_biased_w2v_small.model
            is not gender_biased_w2v_small_copy.model)


def test_evaluate_word_embedding(gender_biased_w2v_small):
    """Test evaluate_word_embedding method in GenderBiasWE."""
    # pylint: disable=C0301
    (word_pairs_evaluation,
     word_analogies_evaluation) = gender_biased_w2v_small.evaluate_word_embedding()

    assert (word_pairs_evaluation.to_dict()
            == {'pearson_r': {'WS353': 0.645, 'RG65': 0.576, 'RW': 0.611, 'Mturk': 0.65, 'MEN': 0.766, 'SimLex999': 0.456, 'TR9856': 0.666},
                'pearson_pvalue': {'WS353': 0.0, 'RG65': 0.232, 'RW': 0.0, 'Mturk': 0.0, 'MEN': 0.0, 'SimLex999': 0.0, 'TR9856': 0.0},
                'spearman_r': {'WS353': 0.688, 'RG65': 0.493, 'RW': 0.655, 'Mturk': 0.674, 'MEN': 0.782, 'SimLex999': 0.444, 'TR9856': 0.676},
                'spearman_pvalue': {'WS353': 0.0, 'RG65': 0.321, 'RW': 0.0, 'Mturk': 0.0, 'MEN': 0.0, 'SimLex999': 0.0, 'TR9856': 0.0},
                'ratio_unkonwn_words': {'WS353': 9.915, 'RG65': 14.286, 'RW': 77.384, 'Mturk': 1.558, 'MEN': 15.148, 'SimLex999': 1.702, 'TR9856': 89.722}})

    assert (word_analogies_evaluation.to_dict()
            == {'score': {'MSR-syntax': 0.75, 'Google': 0.729}})


def test_generate_analogies(gender_biased_w2v_small):
    """Test generate_analogies method in GenderBiasWE.

    Based on:
    https://github.com/tolga-b/debiaswe/blob/master/tutorial_example1.ipynb
    """

    analogies_df = (gender_biased_w2v_small
                    .generate_analogies(500, unrestricted=False))
    analogies_df = analogies_df[['she', 'he']]
    assert_deep_almost_equal(analogies_df.values, TOLGA_GENDER_ANALOGIES)


# TODO deeper testing, this is barely checking it runs
# TODO not all full_specific_words are lower case - why? maybe just names?
# TODO maybe it was trained on the whole w2v?
def test_learn_full_specific_words(gender_biased_w2v_small):
    (full_specific_words,
     clf, X, y) = gender_biased_w2v_small.learn_full_specific_words(debug=True)
    full_specific_words.sort()
    assert (set(gender_biased_w2v_small._data['specific_seed'])
            .issubset(full_specific_words))


def test_calc_all_weat(w2v_small):
    calc_all_weat(w2v_small, filter_by='model', with_original_finding=True,
                  with_pvalue=True, pvalue_kwargs={'method': 'approximate'})


def test_calc_all_weat_index(w2v_small):
    all_weat = calc_all_weat(w2v_small, filter_by='model',
                             with_original_finding=True,
                             with_pvalue=True,
                             pvalue_kwargs={'method': 'approximate'})

    for index in range(len(WEAT_DATA)):
        single_weat = calc_all_weat(w2v_small, weat_data=index,
                                    filter_by='model',
                                    with_original_finding=True,
                                    with_pvalue=True,
                                    pvalue_kwargs={'method':
                                                   'approximate'})

        assert_deep_almost_equal(single_weat.iloc[0].to_dict(),
                                 all_weat.iloc[index].to_dict(),
                                 atol=0.01)


def test_calc_all_weat_indices(w2v_small):
    all_weat = calc_all_weat(w2v_small, filter_by='model',
                             with_original_finding=True,
                             with_pvalue=True,
                             pvalue_kwargs={'method': 'approximate'})

    for index in range(1, len(WEAT_DATA)):
        indices = tuple(range(index))
        singles_weat = calc_all_weat(w2v_small, weat_data=indices,
                                     filter_by='model',
                                     with_original_finding=True,
                                     with_pvalue=True,
                                     pvalue_kwargs={'method':
                                                    'approximate'})

        assert_deep_almost_equal(singles_weat.to_dict(),
                                 all_weat.iloc[:index].to_dict(),
                                 atol=0.01)


def test_calc_all_weat_defaults(w2v_small):
    weat_5_results_default = (calc_all_weat(w2v_small, (5,))
                              .iloc[0].to_dict())

    weat_5_results = (calc_all_weat(w2v_small, (5,),
                                    filter_by='model',
                                    with_pvalue=True,
                                    pvalue_kwargs={'method': 'exact'})
                      .iloc[0].to_dict())

    assert_deep_almost_equal(weat_5_results_default, weat_5_results)


def test_calc_weat_pleasant_attribute(w2v_small):
    # pylint: disable=line-too-long

    pvalue_kwargs = {'method': 'approximate'}

    result_v1 = calc_weat_pleasant_unpleasant_attribute(w2v_small,
                                                        WEAT_DATA[1]['first_target'],
                                                        WEAT_DATA[1]['second_target'],
                                                        pvalue_kwargs=pvalue_kwargs)
    result_v1['p'] = round(result_v1['p'], 4)
    result_v1['d'] = round(result_v1['d'], 4)
    result_v1['s'] = round(result_v1['s'], 4)

    result_v2 = (calc_all_weat(w2v_small, (1,),
                               pvalue_kwargs=pvalue_kwargs)
                 .iloc[0]
                 .to_dict())

    assert_deep_almost_equal(result_v1, result_v2, atol=0.01)


def test_most_similar(w2v_small):
    POSITIVE, NEGATIVE = ('doctor', 'she'), ('he',)

    responsibly_results = most_similar(w2v_small, POSITIVE, NEGATIVE,
                                       topn=10)
    gensim_results = w2v_small.most_similar(POSITIVE, NEGATIVE,
                                            topn=9)

    assert responsibly_results[0][0] == 'doctor'

    assert_deep_almost_equal(responsibly_results[1:], gensim_results,
                             atol=0.01)


def test_compute_association(gender_biased_w2v_small):
    """
    Test compute_association method in GenderBiasWE.
    """
    (r, pvalue), _ = gender_biased_w2v_small.compute_factual_association()

    assert isclose(r, 0.7070401592764508, abs_tol=ATOL)
    assert isclose(pvalue, 1.4324502214459908e-06, abs_tol=ATOL)
