# pylint: disable=C0301

import copy
import random

import numpy as np
import pandas as pd

from ..consts import RANDOM_STATE
from .data import WEAT_DATA
from .utils import assert_gensim_keyed_vectors


FILTER_BY_OPTIONS = ['caliskan', 'model']

RESULTS_DF_COLUMNS = ['Target words', 'Attrib. words',
                      's', 'd', 'p', 'Nt', 'Na']


def _calc_association_target_attributes(model, target_word,
                                        first_attribute_words,
                                        second_attribute_words):
    assert_gensim_keyed_vectors(model)

    first_mean = model.n_similarity([target_word],
                                    first_attribute_words).mean()
    second_mean = model.n_similarity([target_word],
                                     second_attribute_words).mean()

    return first_mean - second_mean


def _calc_association_all_targets_attributes(model, target_words,
                                             first_attribute_words,
                                             second_attribute_words):
    return [_calc_association_target_attributes(model, target_word,
                                                first_attribute_words,
                                                second_attribute_words)
            for target_word in target_words]


def _calc_weat_pvalue():
    return 0.


def calc_single_weat(model,
                     first_target, second_target,
                     first_attribute, second_attribute,
                     with_pvalue=True):

    assert len(first_target['words']) == len(second_target['words'])
    assert len(first_attribute['words']) == len(second_attribute['words'])

    first_associations = _calc_association_all_targets_attributes(model,
                                                                  first_target['words'],
                                                                  first_attribute['words'],
                                                                  second_attribute['words'])

    second_associations = _calc_association_all_targets_attributes(model,
                                                                   second_target['words'],
                                                                   first_attribute['words'],
                                                                   second_attribute['words'])

    score = sum(first_associations) - sum(second_associations)
    std_dev = np.std(first_associations + second_associations)
    effect_size = ((np.mean(first_associations) - np.mean(second_associations))
                   / std_dev)

    pvalue = None
    if with_pvalue:
        pvalue = _calc_weat_pvalue()

    return {'Target words': '{} vs. {}'.format(first_target['name'],
                                               second_target['name']),
            'Attrib. words': '{} vs. {}'.format(first_attribute['name'],
                                                second_attribute['name']),
            's': score,
            'd': effect_size,
            'p': pvalue,
            'Nt': '{}x2'.format(len(first_target['words'])),
            'Na': '{}x2'.format(len(first_attribute['words']))}


def _filter_by_caliskan_weat_stimuli(stimuli):
    """Inplace."""
    for group in stimuli:
        if 'remove' in stimuli[group]:
            words_to_remove = stimuli[group]['remove']
            stimuli[group]['word'] = [word for word in stimuli[group]['words']
                                      if word not in words_to_remove]


def _sample_if_bigger(seq, length):
    if len(seq) > length:
        seq = random.sample(seq, length)
    return seq


def _filter_by_model_weat_stimuli(stimuli, model):
    """Inplace."""
    random.seed(RANDOM_STATE)
    for group_category in ['target', 'attribute']:
        first_group = 'first_' + group_category
        second_group = 'second_' + group_category

        first_words = [word for word in stimuli[first_group]['words']
                       if word in model]
        second_words = [word for word in stimuli[second_group]['words']
                        if word in model]

        min_len = min(len(first_words), len(second_words))

        first_words = _sample_if_bigger(first_words, min_len)
        second_words = _sample_if_bigger(second_words, min_len)

        first_words.sort()
        second_words.sort()

        stimuli[first_group]['words'] = first_words
        stimuli[second_group]['words'] = second_words


def filter_weat_data(weat_data, model, filter_by):
    """inplace."""
    if filter_by not in FILTER_BY_OPTIONS:
        raise ValueError('filter_by should be one of {}, {} was given'.format(
            FILTER_BY_OPTIONS, filter_by))

    if filter_by == 'caliskan':
        for stimuli in weat_data:
            _filter_by_caliskan_weat_stimuli(stimuli)

    elif filter_by == 'model':
        for stimuli in weat_data:
            _filter_by_model_weat_stimuli(stimuli, model)


def calc_all_weat(model, weat_data='caliskan', filter_by='model'):
    if weat_data == 'caliskan':
        weat_data = WEAT_DATA

    weat_data = copy.deepcopy(weat_data)

    filter_weat_data(weat_data,
                     model,
                     filter_by)

    results = []
    for stimuli in weat_data:
        if all([group['words'] for group in stimuli.values()]):
            result = calc_single_weat(model,
                                      stimuli['first_target'],
                                      stimuli['second_target'],
                                      stimuli['first_attribute'],
                                      stimuli['second_attribute'])
            result['stimuli'] = stimuli
        else:
            print(stimuli)

            results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df[RESULTS_DF_COLUMNS]

    return results_df
