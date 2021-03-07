"""BrainVoyager helper functions."""


def bv_parse_voi(fname):
    """Parse a VOI definition file."""

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
