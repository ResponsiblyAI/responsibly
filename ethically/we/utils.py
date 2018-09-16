import math

import numpy as np
import pandas as pd


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
    model.syn0[model.vocab[word].index] = new_vector
    if model.syn0norm is not None:
        model.syn0norm[model.vocab[word].index] = normalize(new_vector)


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
