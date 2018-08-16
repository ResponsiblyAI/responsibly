import numpy as np


def normalize(v):
    if v.ndim != 1:
        raise ValueError('v should be 1-D, {}-D was given'.format(
            v.ndim))
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def cosine_similarity(v, u):
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
    rejected_vector = v - project_vector(v, u)
    return projected_vector, rejected_vector


def update_word_vector(model, word, new_vector):
    model.syn0[model.vocab[word].index] = new_vector
    if model.syn0norm is not None:
        model.syn0norm[model.vocab[word].index] = normalize(new_vector)


def generate_one_word_forms(word):
    return [word.lower(), word.upper(), word.title()]


def generate_words_forms(words):
    return sum([generate_one_word_forms(word) for word in words], [])
