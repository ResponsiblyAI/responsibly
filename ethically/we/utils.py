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
    """Projecting the vector v onto direction u."""
    return v - project_vector(v, u)
