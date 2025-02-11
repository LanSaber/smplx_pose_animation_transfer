from animation import *

with open("pose_data/processed_quart_val.pkl", 'rb') as f:
    import pickle
    pose_dict_list = pickle.load(f)[1]["poses"]

play_pose_parameters(pose_dict_list, output_dir="3433")
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