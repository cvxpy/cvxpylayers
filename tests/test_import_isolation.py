"""Tests that optional dependency imports are isolated from each other.

Each solver interface should import its optional deps independently so that
e.g. a missing JAX doesn't null out PyTorch imports, and vice versa.
"""

import importlib
import sys
from unittest import mock

import pytest

moreau = pytest.importorskip("moreau")


class TestMoreauImportIsolation:
    """moreau, moreau.jax, moreau.torch must import independently."""

    def _reload_with_blocked(self, blocked_module):
        """Reload moreau_if with one submodule blocked, return snapshot."""
        import cvxpylayers.interfaces.moreau_if as moreau_if_module

        saved = {}
        for key in list(sys.modules):
            if key == blocked_module or key.startswith(blocked_module + "."):
                saved[key] = sys.modules.pop(key)
        try:
            with mock.patch.dict(sys.modules, {blocked_module: None}):
                importlib.reload(moreau_if_module)
                snapshot = {
                    "moreau": moreau_if_module.moreau,
                    "moreau_jax": moreau_if_module.moreau_jax,
                    "moreau_torch": moreau_if_module.moreau_torch,
                }
        finally:
            sys.modules.update(saved)
            importlib.reload(moreau_if_module)
        return snapshot

    def test_moreau_torch_survives_missing_jax(self):
        snap = self._reload_with_blocked("moreau.jax")
        assert snap["moreau"] is not None
        assert snap["moreau_torch"] is not None
        assert snap["moreau_jax"] is None

    def test_moreau_jax_survives_missing_torch_submodule(self):
        snap = self._reload_with_blocked("moreau.torch")
        assert snap["moreau"] is not None
        assert snap["moreau_jax"] is not None
        assert snap["moreau_torch"] is None

    def test_all_none_when_moreau_missing(self):
        """If moreau base is missing, all three should be None."""
        import cvxpylayers.interfaces.moreau_if as moreau_if_module

        saved = {}
        for key in list(sys.modules):
            if key == "moreau" or key.startswith("moreau."):
                saved[key] = sys.modules.pop(key)
        try:
            with mock.patch.dict(
                sys.modules,
                {"moreau": None, "moreau.jax": None, "moreau.torch": None},
            ):
                importlib.reload(moreau_if_module)
                snap = {
                    "moreau": moreau_if_module.moreau,
                    "moreau_jax": moreau_if_module.moreau_jax,
                    "moreau_torch": moreau_if_module.moreau_torch,
                }
        finally:
            sys.modules.update(saved)
            importlib.reload(moreau_if_module)
        assert snap["moreau"] is None
        assert snap["moreau_jax"] is None
        assert snap["moreau_torch"] is None


class TestMpaxImportIsolation:
    """jax and mpax must import independently in mpax_if.

    Tests the fix for the same import cascade bug that was in moreau_if.py.
    """

    def test_jax_survives_missing_mpax(self):
        """Even if mpax is not installed, jax should still be importable."""
        pytest.importorskip("jax")

        import cvxpylayers.interfaces.mpax_if as mpax_if_module

        saved = {}
        for key in list(sys.modules):
            if key == "mpax" or key.startswith("mpax."):
                saved[key] = sys.modules.pop(key)
        try:
            with mock.patch.dict(sys.modules, {"mpax": None}):
                importlib.reload(mpax_if_module)
                snap_jax = mpax_if_module.jax
                snap_mpax = mpax_if_module.mpax
        finally:
            sys.modules.update(saved)
            importlib.reload(mpax_if_module)
        assert snap_jax is not None, "jax should still be available when mpax is missing"
        assert snap_mpax is None

    def test_mpax_survives_missing_jax(self):
        """If jax is missing, mpax import block shouldn't crash."""
        import cvxpylayers.interfaces.mpax_if as mpax_if_module

        saved = {}
        for key in list(sys.modules):
            if key.startswith("jax"):
                saved[key] = sys.modules.pop(key)
        try:
            block = {k: None for k in saved}
            block["jax"] = None
            with mock.patch.dict(sys.modules, block):
                importlib.reload(mpax_if_module)
                snap_jax = mpax_if_module.jax
                snap_jnp = mpax_if_module.jnp
        finally:
            sys.modules.update(saved)
            importlib.reload(mpax_if_module)
        assert snap_jax is None
        assert snap_jnp is None
