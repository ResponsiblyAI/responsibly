import math

import gensim
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from six import string_types
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


def cosine_similarities_by_words(model, word, words):
    """Compute cosine similarities between a word and a set of other words."""

    assert isinstance(word, string_types), \
        'The arguemnt `word` should be a string.'
    assert not isinstance(words, string_types), \
        'The argument `words` should not be a string.'

    vec = model[word]
    vecs = [model[w] for w in words]
    return model.cosine_similarities(vec, vecs)


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
        type_names = (model_type.__name__
                      for model_type in WORD_EMBEDDING_MODEL_TYPES)
        raise TypeError('model should be on of the types'
                        ' ({}), not {}.'
                        .format(', '.join(type_names),
                                type(model)))


def most_similar(model, positive=None, negative=None,
                 topn=10, restrict_vocab=None, indexer=None,
                 unrestricted=True):
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
    :param int restrict_vocab: Optional integer which limits the
                               range of vectors
                               which are searched for most-similar values.
                               For example, restrict_vocab=10000 would
                               only check the first 10000 word vectors
                               in the vocabulary order. (This may be
                               meaningful if you've sorted the vocabulary
                               by descending frequency.)
    :param bool unrestricted: Whether to restricted the most
                              similar words to be not from
                              the positive or negative word list.
    :return: Sequence of (word, similarity).
    """
    if topn is not None and topn < 1:
        return []

    if positive is None:
        positive = []
    if negative is None:
        negative = []

    model.init_sims()

    if (isinstance(positive, string_types)
            and not negative):
        # allow calls like most_similar('dog'),
        # as a shorthand for most_similar(['dog'])
        positive = [positive]

    if ((isinstance(positive, string_types) and negative)
            or (isinstance(negative, string_types) and positive)):
        raise ValueError('If positives and negatives are given, '
                         'both should be lists!')

    # add weights for each word, if not already present;
    # default to 1.0 for positive and -1.0 for negative words
    positive = [
        (word, 1.0) if isinstance(word, string_types + (np.ndarray,))
        else word
        for word in positive
    ]
    negative = [
        (word, -1.0) if isinstance(word, string_types + (np.ndarray,))
        else word
        for word in negative
    ]

    # compute the weighted average of all words
    all_words, mean = set(), []
    for word, weight in positive + negative:
        if isinstance(word, np.ndarray):
            mean.append(weight * word)
        else:
            mean.append(weight * model.word_vec(word, use_norm=True))
            if word in model.vocab:
                all_words.add(model.vocab[word].index)

    if not mean:
        raise ValueError("Cannot compute similarity with no input.")
    mean = gensim.matutils.unitvec(np.array(mean)
                                   .mean(axis=0)).astype(float)

    if indexer is not None:
        return indexer.most_similar(mean, topn)

    limited = (model.vectors_norm if restrict_vocab is None
               else model.vectors_norm[:restrict_vocab])
    dists = limited @ mean

    if topn is None:
        return dists

    best = gensim.matutils.argsort(dists,
                                   topn=topn + len(all_words),
                                   reverse=True)

    # if not unrestricted, then ignore (don't return)
    # words from the input
    result = [(model.index2word[sim], float(dists[sim]))
              for sim in best
              if unrestricted or sim not in all_words]

    return result[:topn]


def get_seed_vector(seed, bias_word_embedding):

    if seed == 'direction':
        positive_end = bias_word_embedding.positive_end
        negative_end = bias_word_embedding.negative_end
        bias_word_embedding._is_direction_identified()  # pylint: disable=protected-access
        seed_vector = bias_word_embedding.direction
    else:
        if seed == 'ends':
            positive_end = bias_word_embedding.positive_end
            negative_end = bias_word_embedding.negative_end

        else:
            positive_end, negative_end = seed

        seed_vector = normalize(bias_word_embedding.model[positive_end]
                                - bias_word_embedding.model[negative_end])

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
