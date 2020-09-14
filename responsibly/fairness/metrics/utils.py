import pandas as pd
from sklearn.metrics._classification import _check_targets


def _assert_binary(y1, y2=None):

    if y2 is None:
        y2 = y1

    y_type, _, _ = _check_targets(y1, y2)

    if y_type != 'binary':
        raise ValueError('y_true and y_pred must be binary.')


def _groupby_y_x_sens(y_true, y_score, x_sens):
    return (pd.DataFrame({'y_true': y_true,
                          'y_score': y_score,
                          'x_sens': x_sens})
            .groupby('x_sens'))
