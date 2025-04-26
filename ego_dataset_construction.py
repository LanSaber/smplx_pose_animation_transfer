import copy
import gzip
import os.path

import numpy as np
import pickle

from animation import *
from scipy.spatial.transform import Rotation as R
import pandas as pd
mano_right_data = pd.read_pickle('data/mano/MANO_RIGHT.pkl')
mano_left_data = pd.read_pickle('data/mano/MANO_LEFT.pkl')
hands_mean_right = pd.read_pickle('data/mano/MANO_RIGHT.pkl')['hands_mean']
hands_mean_left = pd.read_pickle('data/mano/MANO_LEFT.pkl')['hands_mean']

with gzip.open(os.path.join("data", "labels_clean.all"), 'rb') as f:
    pose_dicts = pickle.load(f)
    keys = list(pose_dicts.keys())
    for file_id in range(len(keys)):
        file_name = str(file_id)
        pose_dict_array = pose_dicts[keys[file_id]]["smpl"]
        frame_num = pose_dict_array["smplx_root_pose"].shape[0]
        rend_data = []
        for i in range(frame_num):
            pose_dict = {}
            pose_dict["smplx_lhand_pose"] = pose_dict_array["smplx_lhand_pose"][i]
            pose_dict["smplx_rhand_pose"] = pose_dict_array["smplx_rhand_pose"][i]
            pose_dict["smplx_root_pose"] = pose_dict_array["smplx_root_pose"][i]
            pose_dict["smplx_body_pose"] = pose_dict_array["smplx_body_pose"][i]
            pose_dict["smplx_jaw_pose"] = pose_dict_array["smplx_jaw_pose"][i]
            rend_data.append(pose_dict)
        save_pose_into_videos(rend_data, output_dir="vid_ego", file_name=file_name)
        print(f"{file_id} has been done")