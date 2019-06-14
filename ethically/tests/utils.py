import numpy as np


def assert_deep_almost_equal(expected, actual, *args, **kwargs):
    """
    Assert that two complex structures have almost equal contents.

    Compares lists, dicts and tuples recursively. Checks numeric values
    using test_case's :py:meth:`unittest.TestCase.assertAlmostEqual` and
    checks all other values with :py:meth:`unittest.TestCase.assertEqual`.
    Accepts additional positional and keyword arguments and pass those
    intact to assertAlmostEqual() (that's how you specify comparison
    precision).

    https://github.com/larsbutler/oq-engine/blob/master/tests/utils/helpers.py
    """
    # is_root = not '__trace' in kwargs
    # trace = kwargs.pop('__trace', 'ROOT')
    try:
        if isinstance(expected, (int, float, complex,
                                 np.float16, np.float32,
                                 np.float64)):
            np.testing.assert_allclose(expected, actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, np.ndarray)):
            assert len(expected) == len(actual)
            for index, _ in enumerate(expected):
                v1, v2 = expected[index], actual[index]
                assert_deep_almost_equal(v1, v2,
                                         # __trace=repr(index),
                                         *args, **kwargs)
        elif isinstance(expected, dict):
            assert set(expected) == set(actual)
            for key in expected:
                assert_deep_almost_equal(expected[key], actual[key],
                                         # __trace=repr(key),
                                         *args, **kwargs)
        else:
            assert expected == actual
    except AssertionError as exc:
        # exc.__dict__.setdefault('traces', []).append(trace)
        # if is_root:
        #     trace = ' -> '.join(reversed(exc.traces))
        #     exc = AssertionError('{}\nTRACE: {}'
        #                          .format(exc.message, trace))
        raise exc
