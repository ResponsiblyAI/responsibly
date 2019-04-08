from sklearn.metrics.classification import _check_targets


def _assert_binary(y1, y2=None):

    if y2 is None:
        y2 = y1

    y_type, _, _ = _check_targets(y1, y2)

    if y_type != 'binary':
        raise ValueError('y_true and y_pred must be binary.')
