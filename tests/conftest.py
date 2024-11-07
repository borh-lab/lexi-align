import pytest


def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
