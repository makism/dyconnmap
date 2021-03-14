import sys

sys.path.append("/home/makism/Github/dyconnmap-feature_dataset/")
from dyconnmap.future import bv_parse_vtc

import numpy as np

if __name__ == "__main__":
    # voi_fname = (
    #     "/home/makism/Documents/BrainVoyager/SampleData/NF Pilot Study/s01/s01_V5.voi"
    # )
    # voi_desc, vois = bv_parse_voi(fname=voi_fname)
    # arr = np.array(vois[0]["Coords"])
    # print(arr.shape)
    #
    # sys.exit(0)

    # transf_coords = 128.0 - arr
    # transf_coords = transf_coords.astype(np.int32)
    # tmp = transf_coords[:, 0].copy()
    # transf_coords[:, 0], transf_coords[:, 1] = (
    #     transf_coords[:, 1],
    #     transf_coords[:, 2],
    # )
    # transf_coords[:, 2] = tmp
    #
    # np.min(transf_coords)
    # np.max(transf_coords)
    #
    # np.shape(transf_coords)

    # print(transf_coords)

    vtc_fname = "/home/makism/Documents/BrainVoyager/SampleData/NF Pilot Study/s01/sess1/NFC_Run1/connectivity_pilot_SCCTBL_3DMCTS_THPGLMF2c_TAL.vtc"
    metadata, tc = bv_parse_vtc(vtc_fname)
