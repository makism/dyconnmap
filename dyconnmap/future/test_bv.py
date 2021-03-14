"""Test suite for BrainVoyager help functions."""

import pathlib
import os

from .bv import bv_convert_coords, bv_parse_voi, bv_parse_vtc


def test_bv_convert_coords():
    """Test convert coords (BV)."""

    # Disable test on Travis.
    if "TRAVIS" in os.environ:
        assert True


def test_bv_parse_voi1():
    """Test parsing VOI (BV)."""

    # Disable test on Travis.
    if "TRAVIS" in os.environ:
        assert True


def test_bv_parse_vtc1():
    """Test parsing (dryrun) VTC."""

    # Disable test on Travis.
    if "TRAVIS" in os.environ:
        assert True

    else:
        expected_metadata = {
            "x_start": 57,
            "x_end": 231,
            "y_start": 52,
            "y_end": 172,
            "z_start": 59,
            "z_end": 197,
            "num_volumes": 530,
            "data_type": 2,
            "vmr_resolution": 3,
            "ref_space_flag": 3,
            "num_protos": 1,
            "tr": 1000,
            "abs_dim_x": 46,
            "abs_dim_y": 40,
            "abs_dim_z": 58,
        }

        base_dir = (
            pathlib.Path.home() / "Github/dyconnmap-public-master/bv_data/1/s01/"
        ).resolve()
        vtc_fname = f"{base_dir}/sess1/NFC_Run1/connectivity_pilot_SCCTBL_3DMCTS_THPGLMF2c_TAL.vtc"

        vtc_metadata, vtc = bv_parse_vtc(vtc_fname, dryrun=True)

        assert vtc_metadata["x_start"] == expected_metadata["x_start"]
        assert vtc_metadata["x_end"] == expected_metadata["x_end"]
        assert vtc_metadata["y_start"] == expected_metadata["y_start"]
        assert vtc_metadata["y_end"] == expected_metadata["y_end"]
        assert vtc_metadata["z_start"] == expected_metadata["z_start"]
        assert vtc_metadata["z_end"] == expected_metadata["z_end"]
        assert vtc_metadata["num_volumes"] == expected_metadata["num_volumes"]
        assert vtc_metadata["data_type"] == expected_metadata["data_type"]
        assert vtc_metadata["vmr_resolution"] == expected_metadata["vmr_resolution"]
        assert vtc_metadata["ref_space_flag"] == expected_metadata["ref_space_flag"]
        assert vtc_metadata["num_protos"] == expected_metadata["num_protos"]
        assert vtc_metadata["tr"] == expected_metadata["tr"]
