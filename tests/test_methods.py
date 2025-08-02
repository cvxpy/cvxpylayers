"""Sample python file for testing functions from the source code."""

from src.package.hello_world import hello_world


def test_hello() -> None:
    """Run a simple test that can use a mock to override online functionality.

    unit_test_mocks: Fixture located in conftest.py, implicitly imported via
    pytest.
    """
    out = hello_world(0)
    assert out == "string-0", "Expected output did not match"


