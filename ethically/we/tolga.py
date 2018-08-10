import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA

import matplotlib.pylab as plt
import seaborn as sns

from .data import TOLGA_DATA
from .utils import cosine_similarity, normalize, reject_vector


DIRECTION_METHODS = ['single', 'sum', 'pca']
DEBIAS_METHODS = ['hard', 'soft']
FIRST_PC_THRESHOLD = 0.5


class WordsEmbedding:

    def __init__(self, model):
        if not isinstance(model, KeyedVectors):
            raise TypeError('model should be of type KeyedVectors, not {}'
                             .format(type(model)))

        self.model = model

        self.direction = None
        self.positive_end = None
        self.negative_end = None


    def _is_direction_identified(self):
        if self.direction is None:
            raise RuntimeError('The direction was not identified'
                               ' for this {} instance'
                               .format(self.__class__.__name__))

    # There is a mistake in the article
    # it is written (section 5.1):
    # "To identify the gender subspace, we took the ten gender pair difference vectors and computed its principal components (PCs)"
    # however in the source code:
    # https://github.com/tolga-b/debiaswe/blob/10277b23e187ee4bd2b6872b507163ef4198686b/debiaswe/we.py#L235-L245
    def _identify_subspace_by_pca(self, definitional_pairs, n_components):
        matrix = []

        for word1, word2 in definitional_pairs:
            vector1 = normalize(self.model[word1])
            vector2 = normalize(self.model[word2])

            center = (vector1 + vector2) / 2

            matrix.append(vector1 - center)
            matrix.append(vector2 - center)

        pca = PCA(n_components=n_components)
        pca.fit(matrix)

        return pca


    def _identify_direction(self, positive_end, negative_end,
                            definitional, method='pca'):
        if method not in DIRECTION_METHODS:
            raise ValueError('method should be one of {}, {} was given'.format(
                DIRECTION_METHODS, method))


        if method == 'single':
            direction = normalize(normalize(self.model[definitional[0]])
                                  - normalize(self.model[definitional[1]]))


        elif method == 'sum':
            groups = list(zip(*definitional))

            group1_sum_vector = np.sum([self.model[word] for word in groups[0]], axis=0)
            group2_sum_vector = np.sum([self.model[word] for word in groups[1]], axis=0)

            diff_vector = normalize(group1_sum_vector) - normalize(group2_sum_vector)

            direction = normalize(diff_vector)

        elif method == 'pca':
            pca = self._identify_subspace_by_pca(definitional, 1)
            if pca.explained_variance_ratio_[0] < FIRST_PC_THRESHOLD:
                raise RuntimeError('The Explained variance of the first principal component should be at least {}, but it is {}'.
                                   format(FIRST_PC_THRESHOLD, pca.explained_variance_ratio_[0]))
            direction = pca.components_[0]

        self.direction = direction
        self.positive_end = positive_end
        self.negative_end = negative_end


    def project_on_direction(self, word):
        self._is_direction_identified()

        vector = self.model[word]
        projection_score = self.model.cosine_similarities(self.direction,
                                                          [vector])[0]
        return projection_score


    def _calc_projection_scores(self, words):
        self._is_direction_identified()

        df = pd.DataFrame({'word': words})

        # TODO: maybe using cosine_similarities on all the vectors?
        # it might be faster
        df['projection'] = df['word'].apply(self.project_on_direction)
        df = df.sort_values('projection', ascending=False)

        return df


    def plot_projection_scores(self, words, ax=None, axis_projection_step=None):
        self._is_direction_identified()

        projections_df = self._calc_projection_scores(words)
        projections_df['projection'] = projections_df['projection'].round(2)

        if ax is None:
            fig = plt.subplots(1)

        if axis_projection_step is None:
            axis_projection_step = 0.1

        cmap = plt.get_cmap('RdBu')
        projections_df['color'] = ((projections_df['projection'] + 1/2)).apply(cmap)

        most_extream_projection = projections_df['projection'].abs().max().round(1)

        sns.barplot(x='projection', y='word', data=projections_df,
                    palette=projections_df['color'])

        plt.xticks(np.arange(-most_extream_projection, most_extream_projection,
                             axis_projection_step))
        plt.title('← {} {} {} →'.format(self.negative_end,
                                        ' ' * 20,
                                        self.positive_end))

        plt.xlabel('Direction Projection')
        plt.ylabel('Words')


    def calc_direct_bias(self, neutral_words, c=None):
        if c is None:
            c = 1

        projections = self._calc_projection_scores(neutral_words)['projection']
        direct_bias_terms = np.abs(projections) ** c
        direct_bias = direct_bias_terms.sum() / len(neutral_words)

        return direct_bias


    def calc_indirect_bias(self, word1, word2):
        self._is_direction_identified()

        vector1 = normalize(self.model[word1])
        vector2 = normalize(self.model[word2])

        perpendicular_vector1 = reject_vector(vector1, self.direction)
        perpendicular_vector2 = reject_vector(vector2, self.direction)

        inner_product = vector1 @ vector2
        perpendicular_similarity = cosine_similarity(perpendicular_vector1, perpendicular_vector2)

        indirect_bias = (inner_product - perpendicular_similarity) / inner_product
        return indirect_bias

    def debias(self, method='hard'):
        raise NotImplementedError

    def evaluate_words_embedding(self):
        # self.model.evaluate_word_pairs
        # self.model.evaluate_word_analogies
        raise NotImplementedError


class GenderBiasWE(WordsEmbedding):
    PROFESSIONS_NAME = TOLGA_DATA['gender']['professions_names']
    DEFINITIONAL_PAIRS = TOLGA_DATA['gender']['definitional_pairs']

    def __init__(self, model):
        super().__init__(model)
        self._identify_direction('she', 'he',
                                 self.__class__.DEFINITIONAL_PAIRS,
                                 'pca')

    def calc_direct_bias(self, neutral_words='professions', c=None):
        if isinstance(neutral_words, str) and neutral_words == 'professions':
            return super().calc_direct_bias(
                self.__class__.PROFESSIONS_NAME, c)
        else:
            return super().calc_direct_bias(neutral_words)

class RaceBiasWE(WordsEmbedding):
    def __init__(self, model):
        raise NotImplementedError
