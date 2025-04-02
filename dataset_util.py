import os.path
import pickle
import json

def transfer_phonenix2render(file_name, index):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        keys = list(data.keys())
        key_name = keys[index]
        keypoints = data[key_name]
        keypoints_render = []
        for keypoint_frame in keypoints:
            keypoint_dict = {}
            keypoint_dict['smplx_root_pose'] = keypoint_frame[0]['result']["global_orient"].squeeze()
            keypoint_dict['smplx_shape'] = keypoint_frame[0]['result']["betas"].squeeze()
            keypoint_dict['smplx_body_pose'] = keypoint_frame[1]["body_pose_rot"].squeeze()
            keypoint_dict['smplx_lhand_pose'] = keypoint_frame[1]["left_hand_pose_rot"].squeeze()
            keypoint_dict['smplx_rhand_pose'] = keypoint_frame[1]["right_hand_pose_rot"].squeeze()
            keypoint_dict['smplx_jaw_pose'] = keypoint_frame[0]['result']["jaw_pose"].squeeze()
            keypoint_dict['smplx_expr'] = keypoint_frame[0]['result']["expression"].squeeze()
            keypoint_dict['cam_trans'] = keypoint_frame[0]['result']["camera_translation"].squeeze()
            keypoints_render.append(keypoint_dict)
        return key_name, keypoints_render


def transfer_phonenix2json(file_name, index):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        keys = list(data.keys())
        key_name = keys[index]
        keys = list(data.keys())
        key_name = keys[index]
        keypoints = data[key_name]["keypoints"]
        openpose_data = {}
        openpose_data["people"] = []
        keypoint_dict ={}
        for i, keypoint in enumerate(keypoints):
            openpose_data = {}
            openpose_data["people"] = []
            keypoint_dict = {}
            keypoint_dict["pose_keypoints_2d"] = keypoint[:25].reshape((-1)).tolist()
            keypoint_dict["hand_left_keypoints_2d"] = keypoint[91:112].reshape((-1)).tolist()
            keypoint_dict["hand_right_keypoints_2d"] = keypoint[112:133].reshape((-1)).tolist()
            openpose_data["people"].append(keypoint_dict)
            dirc = os.path.join("data", "keypoints", key_name.split("/")[-1])
            if not os.path.exists(dirc):
                os.makedirs(dirc)
            file_path = os.path.join(dirc, str(i)+".json")
            with open(file_path, 'w') as f:
                json.dump(openpose_data, f, indent=4)



if __name__ == '__main__':
    a = 1
    # transfer_phonenix2render('data/phoenix_000000_001500.pkl', 0)
    # transfer_phonenix2json(os.path.join("..", "SLRT", "Spoken2Sign", "data", "phoenix-2014t-keypoints.pkl"), 0)