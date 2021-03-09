"""BrainVoyager helper functions."""

import numpy as np
import struct
from typing import Union, List, Optional, Dict


def bv_convert_coords(coords: List, from_ref_space: str) -> Optional["np.ndarray"]:
    """Convert Coordinates from/to TAL/MNI/BV system.

    Parameters
    ----------
    coords : array-like, shape(n_voxels, 3)
        Input coordinates.

    from_ref_space : string
        Original reference space. Valid options include:
        - TAL
        - MNI

    Returns
    -------
    transf_coords : array-like, shape(n_voxels, 3)
        The transformed coordinates. It will return `None` in case of error.
    """
    transf_coords = None

    arr = np.array(coords)

    if np.atleast_2d(arr):
        if arr.shape[1] != 3:
            return transf_coords

        if from_ref_space == "TAL" or from_ref_space == "MNI":
            transf_coords = 128.0 - arr
            transf_coords = transf_coords.astype(np.int32)
            tmp = transf_coords[:, 0].copy()
            transf_coords[:, 0], transf_coords[:, 1] = (
                transf_coords[:, 1],
                transf_coords[:, 2],
            )
            transf_coords[:, 2] = tmp
        else:
            transf_coords = arr

    return transf_coords


def bv_parse_vtc(fname: str, swapaxes: bool = False) -> Union[Dict, "np.ndarray"]:
    """Parse a VTC file.

    Parameters
    ----------
    fname : string
        Input VTC filename.

    swapaxes : bool
        If `True`, the resulting array will swap the axes Z and X.


    Returns
    -------
    metadata : dict
        A dictionary holding the relevant metadata.

    tc : array-like, shape(len_z, len_y, len_x, n_volumes)
        The extracted Timecourses. The axes `len_z` and `len_x` may be optionaly swapped using the parameter `swapaxes`.

    Notes
    -----
    For the time-being, only the 3rd version of the format is supported.
    """

    metadata = {
        "version": 3,
        "fmr_filename": "",
        "num_protos": 1,
        "protos_filenames": list(),
        "current_protocol": 0,
        "data_type": 2,  # 1 -> short int, 2 -> float
        "num_volumes": 0,
        "vmr_resolution": 3,
        "x_start": 57,
        "x_end": 231,
        "y_start": 52,
        "y_end": 172,
        "z_start": 59,
        "z_end": 197,
        "lr": None,  # 1 -> left-is-right, 2 -> left-is-left, 0 -> unknown
        "ref_space_flag": None,  # 1 -> native, 2 -> ACPC, 3 -> Talarach, 0 -> Unknown
        "tr": None,  # in MS
    }

    tc = None

    with open(fname, "rb") as fp:
        contents = fp.read()

        # Version
        buf = struct.unpack("h", contents[0:2])[0]
        metadata["version"] = buf

        if buf != 3:
            return metadata

        # FMR filename
        fmr_filename = ""
        idx = 2
        while True:
            buf = struct.unpack("c", contents[idx : idx + 1])[0]
            char = buf.decode("utf-8")
            if char == "\0":
                idx += 1
                break
            metadata["fmr_filename"] += char
            idx += 1

        # Number of Protocols attached
        buf = struct.unpack("h", contents[idx : idx + 2])[0]
        metadata["num_protos"] = buf

        # Protocol Filenames
        idx += 2
        for i in range(metadata["num_protos"]):
            proto_fname = ""
            while True:
                buf = struct.unpack("c", contents[idx : idx + 1])[0]
                char = buf.decode("utf-8")
                if char == "\0":
                    idx += 1
                    break
                proto_fname += char
                idx += 1
            metadata["protos_filenames"].append(proto_fname)

        # Current Protocol, Data Type, Number of Volumes
        fields = struct.unpack("3h", contents[idx : idx + 6])
        idx += 6
        metadata["current_protocol"], metadata["data_type"], metadata[
            "num_volumes"
        ] = fields

        # X,Y,Z start and end
        fields = struct.unpack("6h", contents[idx : idx + 12])
        idx += 12
        metadata["x_start"], metadata["x_end"], metadata["y_start"], metadata[
            "y_end"
        ], metadata["z_start"], metadata["z_end"] = fields

        # LR convention, Reference Space flag, TR
        fields = struct.unpack("ssf", contents[idx : idx + 8])
        idx += 8
        metadata["lr"] = fields[0]  # TODO: needs decoding
        metadata["ref_space_flag"] = fields[1]  # TODO: needs decoding
        metadata["tr"] = fields[2]

        # Timecourses
        DataType = metadata["data_type"]
        XStart, XEnd = metadata["x_start"], metadata["x_end"]
        YStart, YEnd = metadata["y_start"], metadata["y_end"]
        ZStart, ZEnd = metadata["z_start"], metadata["z_end"]
        VTCResolution = int(metadata["vmr_resolution"])
        NumVolumes = metadata["num_volumes"]

        DimX = int((XEnd - XStart) / VTCResolution)
        DimY = int((YEnd - YStart) / VTCResolution)
        DimZ = int((ZEnd - ZStart) / VTCResolution)

        read_amount = int(DimZ * DimY * DimX * NumVolumes)
        data_size = 4 if DataType == 2 else 2
        data_code = "f" if DataType == 2 else "H"
        tc = struct.unpack(
            f"{read_amount}{data_code}", contents[idx : idx + read_amount * data_size]
        )

        lenX = abs(DimX)
        lenY = abs(DimY)
        lenZ = abs(DimZ)

        tc = np.array(tc)
        tc = np.reshape(tc, [lenZ, lenY, lenX, NumVolumes])

        if swapaxes:
            tc = np.swapaxes(tc, axis1=0, axis2=2)

    return metadata, tc


def bv_parse_voi(fname: str) -> Union[Dict, List[int]]:
    """Parse a VOI definition file.

    Parameters
    ----------
    fname : str
        Input VOI file.

    Returns
    -------
    voi_dec : dict
        A dictionary holding all the metadata.

    vois : list
        A list of VOIs' coordinates.
    """

    voi_desc = {
        "FileVersion": None,
        "ReferenceSpace": None,
        "OriginalVMRResolutionX": None,
        "OriginalVMRResolutionY": None,
        "OriginalVMRResolutionZ": None,
        "OriginalVMROffsetX": None,
        "OriginalVMROffsetY": None,
        "OriginalVMROffsetZ": None,
        "OriginalVMRFramingCubeDim": None,
        "LeftRightConvention": None,
        "SubjectVOINamingConvention": None,
        "NrOfVOIs": None,
        "NrOfVOIVTCs": None,
    }

    vois = list()

    str_fields = set(["ReferenceSpace", "SubjectVOINamingConvention", "NrOfVOIVTCs"])
    int_fields = set(voi_desc) - str_fields

    curr_voi = None
    parsed_vois = 0
    parsed_voi_lines = 0
    with open(fname, "r") as fp:
        for line in fp:
            line = line.strip()

            if len(line) > 0:
                parts = line.split(":")

                key = parts[0]
                val = None
                if key in str_fields:
                    val = str(parts[1])
                elif key in int_fields:
                    val = int(parts[1])

                if val is not None:
                    voi_desc[key] = val
                    continue

                # Individual VOI description
                if key == "NameOfVOI":
                    curr_voi = dict()
                    curr_voi[key] = parts[1].strip()
                    curr_voi["Coords"] = list()
                elif key == "ColorOfVOI":
                    curr_voi[key] = parts[1].split(" ")
                elif key == "NrOfVoxels":
                    curr_voi[key] = int(parts[1])
                else:
                    coords = list(map(lambda x: int(x), line.split(" ")))
                    curr_voi["Coords"].append(coords)

                    parsed_voi_lines += 1
                    if parsed_voi_lines >= curr_voi["NrOfVoxels"]:
                        vois.append(curr_voi)
                        parsed_voi_lines = 0
                        curr_voi = None

    return voi_desc, vois
