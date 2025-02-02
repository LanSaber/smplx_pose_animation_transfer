from animation import *

with open("pose_data/processed_quart_val.pkl", 'rb') as f:
    import pickle
    pose_dict_list = pickle.load(f)[1]["poses"]

play_pose_parameters(pose_dict_list, output_dir_name="2333")

with open("pose_data/processed_quart_val.pkl", 'rb') as f:
    import pickle
    pose_dict_list = pickle.load(f)[2]["poses"]

play_pose_parameters(pose_dict_list, output_dir_name="3334")