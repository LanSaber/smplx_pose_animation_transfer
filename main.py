import copy
import gzip
import os.path

import numpy as np
import pickle
import pandas as pd

# with gzip.open(os.path.join("data", "labels_clean.all"), 'rb') as f:
#     pose_dicts = pickle.load(f)
#     keys = list(pose_dicts.keys())
#     pose_dict_array = pose_dicts[keys[0]]["smpl"]
#     frame_num = pose_dict_array["smplx_root_pose"].shape[0]
#     rend_data = []
#     for i in range(frame_num):
#         pose_dict = {}
#         pose_dict["smplx_lhand_pose"] = pose_dict_array["smplx_lhand_pose"][i]
#         pose_dict["smplx_rhand_pose"] = pose_dict_array["smplx_rhand_pose"][i]
#         pose_dict["smplx_root_pose"] = pose_dict_array["smplx_root_pose"][i]
#         pose_dict["smplx_body_pose"] = pose_dict_array["smplx_body_pose"][i]
#         pose_dict["smplx_jaw_pose"] = pose_dict_array["smplx_jaw_pose"][i]
#         rend_data.append(pose_dict)

mano_right_data = pd.read_pickle('data/mano/MANO_RIGHT.pkl')
mano_left_data = pd.read_pickle('data/mano/MANO_LEFT.pkl')
hands_mean_right = pd.read_pickle('data/mano/MANO_RIGHT.pkl')['hands_mean']
hands_mean_left  = pd.read_pickle('data/mano/MANO_LEFT.pkl')['hands_mean']

# _, pose_dict_list = transfer_phonenix2render("data/phoenix_000000_001500.pkl", 2)
with open(os.path.join("pose_data", "_2FBDaOPYig_1-3-rgb_front_hand_refine.pkl"), "rb") as f:
    hand_data = pickle.load(f)

# with open(os.path.join("pose_data", "output_video_hand_refine.pkl"), "rb") as f:
#     ego_hand_data = pickle.load(f)


from animation import *
from scipy.spatial.transform import Rotation as R

# adapt to the wrist
# with open(os.path.join("pose_data", "1.pkl"), "rb") as f:
#     data = pickle.load(f)
#     set_index = 24
#     # for i in range(48):
#     #     data[i] = copy.deepcopy(data[set_index])
#     for i in range(len(data["smplx_lhand_pose"])):
#         set_index = i
#
#         arr = int(i / 8)
#         arr = 0
#         arr_index = [[0,1,2],
#                      [0,2,1],
#                      [1,0,2],
#                      [1,2,0],
#                      [2,1,0],
#                      [2,0,1]]
#         # neg = i % 8
#         neg = 0
#         neg_index = [
#             [1, 1, 1],
#             [-1, 1, 1],
#             [1, -1, 1],
#             [1, 1, -1],
#             [-1, -1, 1],
#             [1, -1, -1],
#             [-1, 1, -1],
#             [-1, -1, -1],
#         ]
#
#         rotation_matrix_x = np.array([
#             [1, 0, 0],
#             [0, 0, 1],
#             [0, -1, 0]
#         ],dtype=float)
#
#
#         data["smplx_lhand_pose"][i] = hand_data[set_index]["smplx_lhand_pose"]
#         data["smplx_rhand_pose"][i] = hand_data[set_index]["smplx_rhand_pose"]
#         # if i == 0:
#         #     data["smplx_lhand_pose"][i] = ego_hand_data[set_index]["smplx_lhand_pose"]
#         #     data["smplx_rhand_pose"][i] = ego_hand_data[set_index]["smplx_rhand_pose"]
#         #     hand_data[i]["lwrist_global_orient"] = ego_hand_data[set_index]["lwrist_global_orient"]
#         #     hand_data[i]["rwrist_global_orient"] = ego_hand_data[set_index]["rwrist_global_orient"]
#         if hand_data[i].get("lwrist_global_orient") is not None:
#             # pelvis_matrix = np.identity(3)
#             # pelvis_matrix = R.from_rotvec(data["smplx_body_pose"][i][0:3]).as_matrix()
#             spine1_matrix = R.from_rotvec(data["smplx_body_pose"][i][6:9]).as_matrix()
#             spine2_matrix = R.from_rotvec(data["smplx_body_pose"][i][15:18]).as_matrix()
#             spine3_matrix = R.from_rotvec(data["smplx_body_pose"][i][24:27]).as_matrix()
#
#             upper_body_matrix = np.matmul(spine1_matrix, np.matmul(spine2_matrix, spine3_matrix))
#             # upper_body_matrix = np.identity(3)
#
#
#             left_collar_matrix = R.from_rotvec(data["smplx_body_pose"][i][36:39]).as_matrix()
#             right_collar_matrix = R.from_rotvec(data["smplx_body_pose"][i][39:42]).as_matrix()
#             left_shoulder_matrix = R.from_rotvec(data["smplx_body_pose"][i][45:48]).as_matrix()
#             right_shoulder_matrix = R.from_rotvec(data["smplx_body_pose"][i][48:51]).as_matrix()
#             left_elbow_matrix = R.from_rotvec(data["smplx_body_pose"][i][51:54]).as_matrix()
#             right_elbow_matrix = R.from_rotvec(data["smplx_body_pose"][i][54:57]).as_matrix()
#             left_hand_matrix_prefix = np.matmul(upper_body_matrix, np.matmul(left_collar_matrix, np.matmul(left_shoulder_matrix, left_elbow_matrix)))
#             right_hand_matrix_prefix = np.matmul(upper_body_matrix, np.matmul(right_collar_matrix, np.matmul(right_shoulder_matrix, right_elbow_matrix)))
#             left_hand_rot_global_under_mano = np.array(hand_data[set_index]["lwrist_global_orient"])
#             # left_hand_rot_global_under_mano = np.array([0, 0, 0])
#
#             wrist_pose_matrix = R.from_rotvec(left_hand_rot_global_under_mano, degrees=False).as_matrix()
#             wrist_pose_matrix = np.matmul(rotation_matrix_x, wrist_pose_matrix)
#             left_hand_rot_global_under_mano = R.from_matrix(wrist_pose_matrix).as_rotvec(degrees=False)
#
#             left_hand_rot_global = np.array([neg_index[neg][0]* left_hand_rot_global_under_mano[arr_index[arr][0]],
#                                              neg_index[neg][1]*left_hand_rot_global_under_mano[arr_index[arr][1]], neg_index[neg][2]*left_hand_rot_global_under_mano[arr_index[arr][2]]])
#             # left_hand_rot_global[1:3] = - left_hand_rot_global[1:3]
#             # left_hand_rot_global[1] = -left_hand_rot_global[1]
#             # left_hand_rot_global[2] = -left_hand_rot_global[2]
#             left_hand_matrix_global = R.from_rotvec(left_hand_rot_global).as_matrix()
#
#             # left_transform = R.from_euler('y', 180, degrees=True).as_matrix()
#             # mirror = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
#             # left_hand_matrix_global = mirror @ left_transform @ left_hand_matrix_global
#
#
#             # left_hand_matrix_global = np.identity(3)
#
#             right_hand_rot_global_under_mano = np.array(hand_data[set_index]["rwrist_global_orient"])
#
#             wrist_pose_matrix = R.from_rotvec(right_hand_rot_global_under_mano, degrees=False).as_matrix()
#             wrist_pose_matrix = np.matmul(rotation_matrix_x, wrist_pose_matrix)
#             right_hand_rot_global_under_mano = R.from_matrix(wrist_pose_matrix).as_rotvec(degrees=False)
#
#             right_hand_rot_global = np.array([neg_index[neg][0]* right_hand_rot_global_under_mano[arr_index[arr][0]], neg_index[neg][1]*right_hand_rot_global_under_mano[arr_index[arr][1]], neg_index[neg][2]*right_hand_rot_global_under_mano[arr_index[arr][2]]])
#             # right_hand_rot_global[1:3] = - right_hand_rot_global[1:3]
#             # right_hand_rot_global[0] = -right_hand_rot_global[0]
#             # right_hand_rot_global[2] = -right_hand_rot_global[2]
#
#
#             right_hand_matrix_global = R.from_rotvec(right_hand_rot_global).as_matrix()
#
#             right_transform = R.from_euler('y', 180, degrees=True).as_matrix()
#             # right_hand_matrix_global = right_transform @ right_hand_matrix_global
#
#             # right_hand_matrix_global = np.identity(3)
#             left_hand_matrix_rel = np.matmul(np.linalg.inv(left_hand_matrix_prefix), left_hand_matrix_global)
#             right_hand_matrix_rel = np.matmul(np.linalg.inv(right_hand_matrix_prefix), right_hand_matrix_global)
#
#             data["smplx_body_pose"][i][57:60] = R.from_matrix(left_hand_matrix_rel).as_rotvec()
#             data["smplx_body_pose"][i][60:63] = R.from_matrix(right_hand_matrix_rel).as_rotvec()
#             # data["smplx_body_pose"][i][57:60] = hand_data[set_index]["lwrist_global_orient"]
#             # data["smplx_body_pose"][i][60:63] = hand_data[set_index]["rwrist_global_orient"]
#
#     frame_num = data["smplx_lhand_pose"].shape[0]
#     rend_data = []
#     for i in range(frame_num):
#         pose_dict = {}
#         # pose_dict["smplx_lhand_pose"] = pose_dicts["smplx_lhand_pose"][i]
#         # pose_dict["smplx_rhand_pose"] = pose_dicts["smplx_rhand_pose"][i]
#         pose_dict["smplx_lhand_pose"] = hand_data[i]["smplx_lhand_pose"] - hands_mean_left
#         pose_dict["smplx_rhand_pose"] = hand_data[i]["smplx_rhand_pose"] - hands_mean_right
#         pose_dict["smplx_root_pose"] = data["smplx_root_pose"][i]
#         pose_dict["smplx_body_pose"] = data["smplx_body_pose"][i]
#         pose_dict["smplx_jaw_pose"] = data["smplx_jaw_pose"][i]
#         rend_data.append(pose_dict)


with open("pose_data/_2FBDaOPYig_1-3-rgb_front.pkl", 'rb') as f:
    pose_dicts = pickle.load(f)
    frame_num = pose_dicts["smplx_lhand_pose"].shape[0]
    rend_data = []
    for i in range(frame_num):
        pose_dict = {}
        pose_dict["smplx_lhand_pose"] = pose_dicts["smplx_lhand_pose"][i]
        pose_dict["smplx_rhand_pose"] = pose_dicts["smplx_rhand_pose"][i]
        # pose_dict["smplx_lhand_pose"] = hand_data[i]["smplx_lhand_pose"] - hands_mean_left
        # pose_dict["smplx_rhand_pose"] = hand_data[i]["smplx_rhand_pose"] - hands_mean_right
        pose_dict["smplx_root_pose"] = pose_dicts["smplx_root_pose"][i]
        pose_dict["smplx_body_pose"] = pose_dicts["smplx_body_pose"][i]
        pose_dict["smplx_jaw_pose"] = pose_dicts["smplx_jaw_pose"][i]
        rend_data.append(pose_dict)

play_pose_parameters(rend_data, output_dir="vid1")
#
# with open("pose_data/processed_quart_val.pkl", 'rb') as f:
#     import pickle
#     pose_dict_list = pickle.load(f)[2]["poses"]

# with open("D:\\python project\\fit_body_poses\\output_folder\\results\\_2FBDaOPYig_1-3-rgb_front\\0050_weight.pkl", 'rb') as f:
#     import pickle
#     pose_dict_list = pickle.load(f)[0]["poses"]
#     dawe = 0
#
# play_pose_parameters(pose_dict_list, output_dir_name="3334")