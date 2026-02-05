def pytest_configure(config):
    # Ensure custom markers are registered even if pytest.ini isn't discovered
    config.addinivalue_line(
        "markers", "slow: tests that are slow or require large model downloads"
    )


