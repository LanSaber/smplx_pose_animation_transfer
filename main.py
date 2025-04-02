import copy
import os.path

import numpy as np

from animation import *
import pickle
from scipy.spatial.transform import Rotation as R

# _, pose_dict_list = transfer_phonenix2render("data/phoenix_000000_001500.pkl", 2)
with open(os.path.join("pose_data", "_2FBDaOPYig_1-3-rgb_front_hand_refine.pkl"), "rb") as f:
    hand_data = pickle.load(f)
#
# with open("D:\\python project\\fit_body_poses\\output_folder\\results\\_2FBDaOPYig_1-3-rgb_front\\0125_weight.pkl", "rb") as f:
#     data = pickle.load(f)[0]["poses"]
#     set_index = 24
#     # for i in range(48):
#     #     data[i] = copy.deepcopy(data[set_index])
#     for i in range(len(data)):
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
#         neg = i % 8
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
#         data[i]["smplx_lhand_pose"] = hand_data[set_index]["smplx_lhand_pose"]
#         data[i]["smplx_rhand_pose"] = hand_data[set_index]["smplx_rhand_pose"]
#         if hand_data[i].get("smplx_lwrist_pose") is not None:
#             pelvis_matrix = np.identity(3)
#             # pelvis_matrix = R.from_rotvec(data[i]["smplx_body_pose"][0:3]).as_matrix()
#             spine1_matrix = R.from_rotvec(data[i]["smplx_body_pose"][6:9]).as_matrix()
#             spine2_matrix = R.from_rotvec(data[i]["smplx_body_pose"][15:18]).as_matrix()
#             spine3_matrix = R.from_rotvec(data[i]["smplx_body_pose"][24:27]).as_matrix()
#
#             upper_body_matrix = np.matmul(spine1_matrix, np.matmul(spine2_matrix, spine3_matrix))
#             # upper_body_matrix = np.identity(3)
#
#
#             left_collar_matrix = R.from_rotvec(data[i]["smplx_body_pose"][36:39]).as_matrix()
#             right_collar_matrix = R.from_rotvec(data[i]["smplx_body_pose"][39:42]).as_matrix()
#             left_shoulder_matrix = R.from_rotvec(data[i]["smplx_body_pose"][45:48]).as_matrix()
#             right_shoulder_matrix = R.from_rotvec(data[i]["smplx_body_pose"][48:51]).as_matrix()
#             left_elbow_matrix = R.from_rotvec(data[i]["smplx_body_pose"][51:54]).as_matrix()
#             right_elbow_matrix = R.from_rotvec(data[i]["smplx_body_pose"][54:57]).as_matrix()
#             left_hand_matrix_prefix = np.matmul(upper_body_matrix, np.matmul(left_collar_matrix, np.matmul(left_shoulder_matrix, left_elbow_matrix)))
#             right_hand_matrix_prefix = np.matmul(upper_body_matrix, np.matmul(right_collar_matrix, np.matmul(right_shoulder_matrix, right_elbow_matrix)))
#             left_hand_rot_global_under_mano = np.array(hand_data[set_index]["smplx_lwrist_pose"][0])
#             left_hand_rot_global = np.array([neg_index[neg][0]* left_hand_rot_global_under_mano[arr_index[arr][0]], neg_index[neg][1]*left_hand_rot_global_under_mano[arr_index[arr][1]], neg_index[neg][2]*left_hand_rot_global_under_mano[arr_index[arr][2]]])
#             # left_hand_rot_global[1:3] = - left_hand_rot_global[1:3]
#             # left_hand_rot_global[1] = -left_hand_rot_global[1]
#             left_hand_rot_global[2] = -left_hand_rot_global[2]
#             left_hand_matrix_global = R.from_rotvec(left_hand_rot_global).as_matrix()
#
#             # left_hand_matrix_global = np.identity(3)
#
#             if i == 125:
#                 dwa = 0
#             right_hand_rot_global_under_mano = np.array(hand_data[set_index]["smplx_rwrist_pose"][0])
#             right_hand_rot_global = np.array([neg_index[neg][0]* right_hand_rot_global_under_mano[arr_index[arr][0]], neg_index[neg][1]*right_hand_rot_global_under_mano[arr_index[arr][1]], neg_index[neg][2]*right_hand_rot_global_under_mano[arr_index[arr][2]]])
#             # right_hand_rot_global[1:3] = - right_hand_rot_global[1:3]
#             # right_hand_rot_global[0] = -right_hand_rot_global[0]
#             # right_hand_rot_global[2] = -right_hand_rot_global[2]
#
#
#             right_hand_matrix_global = R.from_rotvec(right_hand_rot_global).as_matrix()
#
#             # right_hand_matrix_global = np.identity(3)
#             left_hand_matrix_rel = np.matmul(np.linalg.inv(left_hand_matrix_prefix), left_hand_matrix_global)
#             right_hand_matrix_rel = np.matmul(np.linalg.inv(right_hand_matrix_prefix), right_hand_matrix_global)
#
#             data[i]["smplx_body_pose"][57:60] = R.from_matrix(left_hand_matrix_rel).as_rotvec()
#             data[i]["smplx_body_pose"][60:63] = R.from_matrix(right_hand_matrix_rel).as_rotvec()
#     with open("combined_pose.pkl", "wb") as fw:
#         pickle.dump(data, fw)

# with open("D:\\python project\\fit_body_poses\\output_folder\\results\\_2FBDaOPYig_1-3-rgb_front\\0125_weight.pkl", 'rb') as f:
#     import pickle
#     pose_dict_list = pickle.load(f)[0]["poses"]
#     data = pose_dict_list


with open("pose_data/1.pkl", 'rb') as f:
    import pickle
    pose_dicts = pickle.load(f)
    frame_num = pose_dicts["smplx_lhand_pose"].shape[0]
    data = []
    for i in range(frame_num):
        pose_dict = {}
        # pose_dict["smplx_lhand_pose"] = pose_dicts["smplx_lhand_pose"][i]
        # pose_dict["smplx_rhand_pose"] = pose_dicts["smplx_rhand_pose"][i]
        pose_dict["smplx_lhand_pose"] = hand_data[i]["smplx_lhand_pose"]
        pose_dict["smplx_rhand_pose"] = hand_data[i]["smplx_rhand_pose"]
        pose_dict["smplx_root_pose"] = pose_dicts["smplx_root_pose"][i]
        pose_dict["smplx_body_pose"] = pose_dicts["smplx_body_pose"][i]
        pose_dict["smplx_jaw_pose"] = pose_dicts["smplx_jaw_pose"][i]
        data.append(pose_dict)

play_pose_parameters(data, output_dir="vid_refine")
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