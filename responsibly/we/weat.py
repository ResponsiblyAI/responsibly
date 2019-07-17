"""
Compute WEAT score of a Word Embedding.

WEAT is a bias measurement method for word embedding,
which is inspired by the `IAT <https://en.wikipedia.org/wiki/Implicit-association_test>`_
(Implicit Association Test) for humans.
It measures the similarity between two sets of *target words*
(e.g., programmer, engineer, scientist, ... and nurse, teacher, librarian, ...)
and two sets of *attribute words* (e.g., man, male, ... and woman, female ...).
A p-value is calculated using a permutation-test.

Reference:
    - Caliskan, A., Bryson, J. J., & Narayanan, A. (2017).
      `Semantics derived automatically
      from language corpora contain human-like biases
      <http://opus.bath.ac.uk/55288/>`_.
      Science, 356(6334), 183-186.

.. important::
    The effect size and pvalue in the WEAT have
    entirely different meaning from those reported in IATs (original finding).
    Refer to the paper for more details.

Stimulus and original finding from:

- [0, 1, 2]
  A. G. Greenwald, D. E. McGhee, J. L. Schwartz,
  Measuring individual differences in implicit cognition:
  the implicit association test.,
  Journal of personality and social psychology 74, 1464 (1998).

- [3, 4]:
  M. Bertrand, S. Mullainathan, Are Emily and Greg more employable
  than Lakisha and Jamal? a field experiment on labor market discrimination,
  The American Economic Review 94, 991 (2004).

- [5, 6, 9]:
  B. A. Nosek, M. Banaji, A. G. Greenwald, Harvesting implicit group attitudes
  and beliefs from a demonstration web site.,
  Group Dynamics: Theory, Research, and Practice 6, 101 (2002).

- [7]:
  B. A. Nosek, M. R. Banaji, A. G. Greenwald, Math=male, me=female,
  therefore math≠me.,
  Journal of Personality and Social Psychology 83, 44 (2002).

- [8]
  P. D. Turney, P. Pantel, From frequency to meaning:
  Vector space models of semantics,
  Journal of Artificial Intelligence Research 37, 141 (2010).
"""

# pylint: disable=C0301

import copy
import random
import warnings

import numpy as np
import pandas as pd
from mlxtend.evaluate import permutation_test

from responsibly.consts import RANDOM_STATE
from responsibly.utils import _warning_setup
from responsibly.we.data import WEAT_DATA
from responsibly.we.utils import (
    assert_gensim_keyed_vectors, cosine_similarities_by_words,
)


FILTER_BY_OPTIONS = ['model', 'data']
RESULTS_DF_COLUMNS = ['Target words', 'Attrib. words',
                      'Nt', 'Na', 's', 'd', 'p']
PVALUE_METHODS = ['exact', 'approximate']
PVALUE_DEFUALT_METHOD = 'exact'
ORIGINAL_DF_COLUMNS = ['original_' + key for key in ['N', 'd', 'p']]
WEAT_WORD_SETS = ['first_target', 'second_target',
                  'first_attribute', 'second_attribute']
PVALUE_EXACT_WARNING_LEN = 10

_warning_setup()


def _calc_association_target_attributes(model, target_word,
                                        first_attribute_words,
                                        second_attribute_words):
    # pylint: disable=line-too-long

    assert_gensim_keyed_vectors(model)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)

        first_mean = (cosine_similarities_by_words(model,
                                                   target_word,
                                                   first_attribute_words)
                      .mean())

        second_mean = (cosine_similarities_by_words(model,
                                                    target_word,
                                                    second_attribute_words)
                       .mean())

    return first_mean - second_mean


def _calc_association_all_targets_attributes(model, target_words,
                                             first_attribute_words,
                                             second_attribute_words):
    return [_calc_association_target_attributes(model, target_word,
                                                first_attribute_words,
                                                second_attribute_words)
            for target_word in target_words]


def _calc_weat_score(model,
                     first_target_words, second_target_words,
                     first_attribute_words, second_attribute_words):

    (first_associations,
     second_associations) = _calc_weat_associations(model,
                                                    first_target_words,
                                                    second_target_words,
                                                    first_attribute_words,
                                                    second_attribute_words)

    return sum(first_associations) - sum(second_associations)


def _calc_weat_pvalue(first_associations, second_associations,
                      method=PVALUE_DEFUALT_METHOD):

    if method not in PVALUE_METHODS:
        raise ValueError('method should be one of {}, {} was given'.format(
            PVALUE_METHODS, method))

    pvalue = permutation_test(first_associations, second_associations,
                              func=lambda x, y: sum(x) - sum(y),
                              method=method,
                              seed=RANDOM_STATE)  # if exact - no meaning
    return pvalue


def _calc_weat_associations(model,
                            first_target_words, second_target_words,
                            first_attribute_words, second_attribute_words):

    assert len(first_target_words) == len(second_target_words)
    assert len(first_attribute_words) == len(second_attribute_words)

    first_associations = _calc_association_all_targets_attributes(model,
                                                                  first_target_words,
                                                                  first_attribute_words,
                                                                  second_attribute_words)

    second_associations = _calc_association_all_targets_attributes(model,
                                                                   second_target_words,
                                                                   first_attribute_words,
                                                                   second_attribute_words)

    return first_associations, second_associations


def _filter_by_data_weat_stimuli(stimuli):
    """Filter WEAT stimuli words if there is a `remove` key.

    Some of the words from Caliskan et al. (2017) seems as not being used.

    Modifiy the stimuli object in place.
    """

    for group in stimuli:
        if 'remove' in stimuli[group]:
            words_to_remove = stimuli[group]['remove']
            stimuli[group]['words'] = [word for word in stimuli[group]['words']
                                       if word not in words_to_remove]


def _sample_if_bigger(seq, length):
    random.seed(RANDOM_STATE)
    if len(seq) > length:
        seq = random.sample(seq, length)
    return seq


def _filter_by_model_weat_stimuli(stimuli, model):
    """Filter WEAT stimuli words if they are not exists in the model.

    Modifiy the stimuli object in place.
    """

    for group_category in ['target', 'attribute']:
        first_group = 'first_' + group_category
        second_group = 'second_' + group_category

        first_words = [word for word in stimuli[first_group]['words']
                       if word in model]
        second_words = [word for word in stimuli[second_group]['words']
                        if word in model]

        min_len = min(len(first_words), len(second_words))

        # sample to make the first and second word set
        # with equal size
        first_words = _sample_if_bigger(first_words, min_len)
        second_words = _sample_if_bigger(second_words, min_len)

        first_words.sort()
        second_words.sort()

        stimuli[first_group]['words'] = first_words
        stimuli[second_group]['words'] = second_words


def _filter_weat_data(weat_data, model, filter_by):
    """inplace."""

    if filter_by not in FILTER_BY_OPTIONS:
        raise ValueError('filter_by should be one of {}, {} was given'.format(
            FILTER_BY_OPTIONS, filter_by))

    if filter_by == 'data':
        for stimuli in weat_data:
            _filter_by_data_weat_stimuli(stimuli)

    elif filter_by == 'model':
        for stimuli in weat_data:
            _filter_by_model_weat_stimuli(stimuli, model)


def calc_single_weat(model,
                     first_target, second_target,
                     first_attribute, second_attribute,
                     with_pvalue=True, pvalue_kwargs=None):
    """
    Calc the WEAT result of a word embedding.

    :param model: Word embedding model of ``gensim.model.KeyedVectors``
    :param dict first_target: First target words list and its name
    :param dict second_target: Second target words list and its name
    :param dict first_attribute: First attribute words list and its name
    :param dict second_attribute: Second attribute words list and its name
    :param bool with_pvalue: Whether to calculate the p-value of the
                             WEAT score (might be computationally expensive)
    :return: WEAT result (score, size effect, Nt, Na and p-value)
    """

    if pvalue_kwargs is None:
        pvalue_kwargs = {}

    (first_associations,
     second_associations) = _calc_weat_associations(model,
                                                    first_target['words'],
                                                    second_target['words'],
                                                    first_attribute['words'],
                                                    second_attribute['words'])

    if first_associations and second_associations:
        score = sum(first_associations) - sum(second_associations)
        std_dev = np.std(first_associations + second_associations, ddof=0)
        effect_size = ((np.mean(first_associations) - np.mean(second_associations))
                       / std_dev)

        pvalue = None
        if with_pvalue:
            pvalue = _calc_weat_pvalue(first_associations,
                                       second_associations,
                                       **pvalue_kwargs)
    else:
        score, std_dev, effect_size, pvalue = None, None, None, None

    return {'Target words': '{} vs. {}'.format(first_target['name'],
                                               second_target['name']),
            'Attrib. words': '{} vs. {}'.format(first_attribute['name'],
                                                second_attribute['name']),
            's': score,
            'd': effect_size,
            'p': pvalue,
            'Nt': '{}x2'.format(len(first_target['words'])),
            'Na': '{}x2'.format(len(first_attribute['words']))}


def calc_weat_pleasant_unpleasant_attribute(model,
                                            first_target, second_target,
                                            with_pvalue=True, pvalue_kwargs=None):
    """
    Calc the WEAT result with pleasent vs. unpleasant attributes.

    :param model: Word embedding model of ``gensim.model.KeyedVectors``
    :param dict first_target: First target words list and its name
    :param dict second_target: Second target words list and its name
    :param bool with_pvalue: Whether to calculate the p-value of the
                             WEAT score (might be computationally expensive)
    :return: WEAT result (score, size effect, Nt, Na and p-value)
    """

    weat_data = {'first_attribute': copy.deepcopy(WEAT_DATA[0]['first_attribute']),
                 'second_attribute': copy.deepcopy(WEAT_DATA[0]['second_attribute']),
                 'first_target': first_target,
                 'second_target': second_target}

    _filter_by_model_weat_stimuli(weat_data, model)

    if pvalue_kwargs is None:
        pvalue_kwargs = {}

    return calc_single_weat(model,
                            **weat_data,
                            with_pvalue=with_pvalue, pvalue_kwargs=pvalue_kwargs)


def calc_all_weat(model, weat_data='caliskan', filter_by='model',
                  with_original_finding=False,
                  with_pvalue=True, pvalue_kwargs=None):
    """
    Calc the WEAT results of a word embedding on multiple cases.

    Note that for the effect size and pvalue in the WEAT have
    entirely different meaning from those reported in IATs (original finding).
    Refer to the paper for more details.

    :param model: Word embedding model of ``gensim.model.KeyedVectors``
    :param dict weat_data: WEAT cases data.
                           - If `'caliskan'` (default) then all
                              the experiments from the original will be used.
                           - If an interger, then the specific experiment by index
                             from the original paper will be used.
                           - If a interger, then tje specific experiments by indices
                             from the original paper will be used.

    :param bool filter_by: Whether to filter the word lists
                           by the `model` (`'model'`)
                           or by the `remove` key in `weat_data` (`'data'`).
    :param bool with_original_finding: Show the origina
    :param bool with_pvalue: Whether to calculate the p-value of the
                             WEAT results (might be computationally expensive)
    :return: :class:`pandas.DataFrame` of WEAT results
             (score, size effect, Nt, Na and p-value)
    """

    if weat_data == 'caliskan':
        weat_data = WEAT_DATA
    elif isinstance(weat_data, int):
        index = weat_data
        weat_data = WEAT_DATA[index:index + 1]
    elif isinstance(weat_data, tuple):
        weat_data = [WEAT_DATA[index] for index in weat_data]

    if (not pvalue_kwargs
            or pvalue_kwargs['method'] == PVALUE_DEFUALT_METHOD):
        max_word_set_len = max(len(stimuli[ws]['words'])
                               for stimuli in weat_data
                               for ws in WEAT_WORD_SETS)
        if max_word_set_len > PVALUE_EXACT_WARNING_LEN:
            warnings.warn('At least one stimuli has a word set bigger'
                          ' than {}, and the computation might take a while.'
                          ' Consider using \'exact\' as method'
                          ' for pvalue_kwargs.'.format(PVALUE_EXACT_WARNING_LEN))

    actual_weat_data = copy.deepcopy(weat_data)

    _filter_weat_data(actual_weat_data,
                      model,
                      filter_by)

    if weat_data != actual_weat_data:
        warnings.warn('Given weat_data was filterd by {}.'
                      .format(filter_by))

    results = []
    for stimuli in actual_weat_data:
        result = calc_single_weat(model,
                                  stimuli['first_target'],
                                  stimuli['second_target'],
                                  stimuli['first_attribute'],
                                  stimuli['second_attribute'],
                                  with_pvalue, pvalue_kwargs)

        # TODO: refactor - check before if one group is without words
        # because of the filtering
        if not all(group['words'] for group in stimuli.values()
                   if 'words' in group):
            result['score'] = None
            result['effect_size'] = None
            result['pvalue'] = None

        result['stimuli'] = stimuli

        if with_original_finding:
            result.update({'original_' + k: v
                           for k, v in stimuli['original_finding'].items()})
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.replace('nan', None)
    results_df = results_df.fillna('')

    # if not results_df.empty:
    cols = RESULTS_DF_COLUMNS[:]
    if with_original_finding:
        cols += ORIGINAL_DF_COLUMNS
    if not with_pvalue:
        cols.remove('p')
    else:
        results_df['p'] = results_df['p'].apply(lambda pvalue: '{:0.1e}'.format(pvalue)  # pylint: disable=W0108
                                                if pvalue else pvalue)

    results_df = results_df[cols]
    results_df = results_df.round(4)

    return results_df
