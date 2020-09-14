"""Unit tests configuration file."""


def pytest_configure(config):
    """Disable verbose output when running tests."""

    terminal = config.pluginmanager.getplugin('terminal')

    class QuietReporter(terminal.TerminalReporter):
        @property
        def verbosity(self):
            return 0

        @property
        def showlongtestinfo(self):
            return False

        @property
        def showfspath(self):
            return False

    terminal.TerminalReporter = QuietReporter
