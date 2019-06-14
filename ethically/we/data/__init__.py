# TODO how import files from a package
import json
import warnings

from gensim.models.keyedvectors import KeyedVectors
from pkg_resources import resource_filename, resource_string


def load_w2v_small():
    """Load reduced Word2Vec model as `KeyedVectors` object.

    Based on the pre-trained embedding on the Google News corpus:
    https://code.google.com/archive/p/word2vec/
    """
    # pylint: disable=C0301

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        model = KeyedVectors.load_word2vec_format(
            resource_filename(__name__, 'GoogleNews-vectors-negative300-bolukbasi.bin'),
            binary=True)

    return model


def load_json_resource(resource_name):
    return json.loads(
        resource_string(__name__, resource_name + '.json').decode('utf-8')

    )


BOLUKBASI_DATA = load_json_resource('bolukbasi')

BOLUKBASI_DATA['gender']['profession_names'] = list(
    zip(*BOLUKBASI_DATA['gender']['professions']))[0]


BOLUKBASI_DATA['gender']['specific_full'].sort()

# TODO: in the code of the article, the last definitional pair
# is not in the specific full
BOLUKBASI_DATA['gender']['specific_full_with_definitional_equalize'] = list(
    (set.union(
        *map(set, BOLUKBASI_DATA['gender']['definitional_pairs']))
     | set.union(
         *map(set, BOLUKBASI_DATA['gender']['equalize_pairs']))
     | set(BOLUKBASI_DATA['gender']['specific_full']))
)
BOLUKBASI_DATA['gender']['specific_full_with_definitional_equalize'].sort()

BOLUKBASI_DATA['gender']['neutral_profession_names'] = list(
    set(BOLUKBASI_DATA['gender']['profession_names'])
    - set(BOLUKBASI_DATA['gender']['specific_full_with_definitional_equalize'])
)
BOLUKBASI_DATA['gender']['neutral_profession_names'].sort()

BOLUKBASI_DATA['gender']['word_group_keys'] = ['profession_names',
                                               'neutral_profession_names',
                                               'specific_seed',
                                               'specific_full',
                                               'specific_full_with_definitional_equalize']  # pylint: disable=C0301


WEAT_DATA = load_json_resource('weat')

# Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K. W. (2018).
# Gender bias in coreference resolution: Evaluation and debiasing methods.
# arXiv preprint arXiv:1804.06876.
# https://arxiv.org/abs/1804.06876
OCCUPATION_FEMALE_PRECENTAGE = load_json_resource(
    'occupational_female_precentage')
