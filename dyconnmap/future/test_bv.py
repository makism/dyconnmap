"""Test suite for BrainVoyager help functions."""

import os

from .bv import bv_convert_coords, bv_parse_voi, bv_parse_vtc


def test_bv_convert_coords():
    """Test convert coords (BV)."""

    # Disable test on Travis.
    if "TRAVIS" in os.environ:
        assert True


def test_bv_parse_voi():
    """Test parsing VOI (BV)."""

    # Disable test on Travis.
    if "TRAVIS" in os.environ:
        assert True


def test_bv_parse_vtc():
    """Test parsing VTC (BV)."""

    # Disable test on Travis.
    if "TRAVIS" in os.environ:
        assert True
