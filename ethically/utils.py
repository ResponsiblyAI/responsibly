import warnings


# https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
def _warning_setup():
    # pylint: disable=line-too-long
    formatwarning_orig = warnings.formatwarning
    warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
        formatwarning_orig(message, category, filename, lineno, line='')
