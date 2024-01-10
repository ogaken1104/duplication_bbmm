# conftest.py


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if exitstatus == 0:
        terminalreporter.write_line("Congratulations! All tests passed successfully.")
