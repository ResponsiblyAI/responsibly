import itertools
import math

import gensim
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score


WORD_EMBEDDING_MODEL_TYPES = (gensim.models.keyedvectors.KeyedVectors,
                              gensim.models.keyedvectors.BaseKeyedVectors,
                              gensim.models.fasttext.FastText,
                              gensim.models.word2vec.Word2Vec,
                              gensim.models.base_any2vec.BaseWordEmbeddingsModel,)  # pylint: disable=line-too-long


def round_to_extreme(value, digits=2):
    place = 10**digits
    new_value = math.ceil(abs(value) * place) / place
    if value < 0:
        new_value = -new_value
    return new_value


def normalize(v):
    """Normalize a 1-D vector."""
    if v.ndim != 1:
        raise ValueError('v should be 1-D, {}-D was given'.format(
            v.ndim))
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def cosine_similarity(v, u):
    """Calculate the cosine similarity between two vectors."""
    v_norm = np.linalg.norm(v)
    u_norm = np.linalg.norm(u)
    similarity = v @ u / (v_norm * u_norm)
    return similarity


def project_vector(v, u):
    """Projecting the vector v onto direction u."""
    normalize_u = normalize(u)
    return (v @ normalize_u) * normalize_u


def reject_vector(v, u):
    """Rejecting the vector v onto direction u."""
    return v - project_vector(v, u)


def project_reject_vector(v, u):
    """Projecting and rejecting the vector v onto direction u."""
    projected_vector = project_vector(v, u)
    rejected_vector = v - projected_vector
    return projected_vector, rejected_vector


def project_params(u, v):
    """Projecting and rejecting the vector v onto direction u with scalar."""
    normalize_u = normalize(u)
    projection = (v @ normalize_u)
    projected_vector = projection * normalize_u
    rejected_vector = v - projected_vector
    return projection, projected_vector, rejected_vector


def update_word_vector(model, word, new_vector):
    model.vectors[model.vocab[word].index] = new_vector
    if model.vectors_norm is not None:
        model.vectors_norm[model.vocab[word].index] = normalize(new_vector)


def generate_one_word_forms(word):
    return [word.lower(), word.upper(), word.title()]


def generate_words_forms(words):
    return sum([generate_one_word_forms(word) for word in words], [])


def take_two_sides_extreme_sorted(df, n_extreme,
                                  part_column=None,
                                  head_value='',
                                  tail_value=''):
    head_df = df.head(n_extreme)[:]
    tail_df = df.tail(n_extreme)[:]

    if part_column is not None:
        head_df[part_column] = head_value
        tail_df[part_column] = tail_value

    return (pd.concat([head_df, tail_df])
            .drop_duplicates()
            .reset_index(drop=True))


def assert_gensim_keyed_vectors(model):
    if not isinstance(model, WORD_EMBEDDING_MODEL_TYPES):
        raise TypeError('model should be of type {}, not {}'
                        .format(''.join(WORD_EMBEDDING_MODEL_TYPES),
                                type(model)))


def most_similar(model, positive=None, negative=None,
                 topn=10, unrestricted=True):
    """
    Find the top-N most similar words.

    Positive words contribute positively towards the similarity,
    negative words negatively.

    This function computes cosine similarity between a simple mean
    of the projection weight vectors of the given words and
    the vectors for each word in the model.
    The function corresponds to the `word-analogy` and `distance`
    scripts in the original word2vec implementation.

    Based on Gensim implementation.

    :param model: Word embedding model of ``gensim.model.KeyedVectors``.
    :param list positive: List of words that contribute positively.
    :param list negative: List of words that contribute negatively.
    :param int topn: Number of top-N similar words to return.
    :param bool unrestricted: Whether to restricted the most
                              similar words to be not from
                              the positive or negative word list.
    :return: Sequence of (word, similarity).
    """

    assert positive is not None or negative is not None, \
           ('At least one of positive or negative arguments'
            ' should be not None.')

    if positive is None:
        positive = []
    elif isinstance(positive, str):
        positive = [positive]

    if negative is None:
        negative = []
    elif isinstance(negative, str):
        negative = [negative]

    positive_vectors = [model[word] for word in positive]
    negative_vectors = [model[word] for word in negative]

    mean_vector = (np.sum(positive_vectors, axis=0)
                   - np.sum(negative_vectors, axis=0))
    mean_vector = normalize(mean_vector)

    cos_distances = model.vectors @ mean_vector

    most_similar_indices = np.argsort(cos_distances)[::-1]

    most_similar_words = (model.index2word[index]
                          for index in most_similar_indices)
    most_similar_distances = (float(cos_distances[index])
                              for index in most_similar_indices)

    most_similar_results = zip(most_similar_words, most_similar_distances)

    if not unrestricted:
        most_similar_results = ((word, distance)
                                for word, distance in most_similar_results
                                if word not in positive
                                and word not in negative)

    return list(itertools.islice(most_similar_results, topn))


def get_seed_vector(seed, bias_words_embedding):

    if seed == 'direction':
        positive_end = bias_words_embedding.positive_end
        negative_end = bias_words_embedding.negative_end
        bias_words_embedding._is_direction_identified()  # pylint: disable=protected-access
        seed_vector = bias_words_embedding.direction
    else:
        if seed == 'ends':
            positive_end = bias_words_embedding.positive_end
            negative_end = bias_words_embedding.negative_end

        else:
            positive_end, negative_end = seed

        seed_vector = normalize(bias_words_embedding.model[positive_end]
                                - bias_words_embedding.model[negative_end])

    return seed_vector, positive_end, negative_end


def plot_clustering_as_classification(X, y_true, random_state=1, ax=None):

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    y_cluster = (KMeans(n_clusters=2, random_state=random_state)
                 .fit_predict(X))

    embedded_vectors = (TSNE(n_components=2, random_state=random_state)
                        .fit_transform(X))

    for y_value in np.unique(y_cluster):
        mask = (y_cluster == y_value)
        label = 'Positive' if y_value else 'Negative'
        ax.scatter(embedded_vectors[mask, 0],
                   embedded_vectors[mask, 1],
                   label=label)

    ax.legend()

    acc = accuracy_score(y_true, y_cluster)

    return max(acc, 1 - acc)
