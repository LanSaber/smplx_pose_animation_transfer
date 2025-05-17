import copy
import gzip
import os.path

import numpy as np
import pickle
import pandas as pd

from scipy.spatial.transform import Rotation as R

from regress import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
left_hand_regressor = ResNetRegression().to(device)
right_hand_regressor = ResNetRegression().to(device)

left_hand_regressor.load_state_dict(torch.load(os.path.join("regressor", "left_hand_global_0_video_500_regressor.pth")))
right_hand_regressor.load_state_dict(torch.load(os.path.join("regressor", "right_hand_global_0_video_500_regressor.pth")))

left_hand_regressor.eval()
right_hand_regressor.eval()

index = 502

input_video_path = os.path.join("render_result", "vid_ego", f"{index}.mp4")

# Open input video
vid = cv2.VideoCapture(input_video_path)

total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float32)

ava_list = []

with gzip.open(os.path.join("data", "labels_clean.all"), 'rb') as f:
    pose_dicts = pickle.load(f)
    keys = list(pose_dicts.keys())
    pose_dict_array = pose_dicts[keys[index]]["smpl"]
    frame_num = pose_dict_array["smplx_root_pose"].shape[0]
    assert frame_num == total_frames
    rend_data = []
    for i in range(frame_num):
        vid.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, img_per_frame = vid.read()

        image = cv2.cvtColor(img_per_frame, cv2.COLOR_BGR2RGB)
        outputs = pipe.predict(image)

        lwrist_pose = None
        rwrist_pose = None

        for output in outputs:
            wilor_pred = output["wilor_preds"]
            hand_pose = torch.from_numpy(wilor_pred["hand_pose"].reshape(1, -1))
            cam_t = torch.from_numpy(wilor_pred['pred_cam_t_full'])
            global_orient = torch.from_numpy(wilor_pred['global_orient'][0])
            input_tensor = torch.cat((global_orient, cam_t, hand_pose), dim=1)
            input_tensor = input_tensor.float().to(device)
            if output["is_right"] == 0.0:
                output_tensor = left_hand_regressor(input_tensor)
                output_tensor = output_tensor.detach().cpu()
                output_rotmat = rot6d_to_rotmat(output_tensor).numpy()
                lwrist_pose = R.from_matrix(output_rotmat).as_rotvec(degrees=False)
            else:
                output_tensor = right_hand_regressor(input_tensor)
                output_tensor = output_tensor.detach().cpu()
                output_rotmat = rot6d_to_rotmat(output_tensor).numpy()
                rwrist_pose = R.from_matrix(output_rotmat).as_rotvec(degrees=False)

        pose_dict = {}
        pose_dict["smplx_lhand_pose"] = pose_dict_array["smplx_lhand_pose"][i]
        pose_dict["smplx_rhand_pose"] = pose_dict_array["smplx_rhand_pose"][i]
        pose_dict["smplx_root_pose"] = pose_dict_array["smplx_root_pose"][i]
        pose_dict["smplx_body_pose"] = pose_dict_array["smplx_body_pose"][i]
        pose_dict["smplx_jaw_pose"] = pose_dict_array["smplx_jaw_pose"][i]

        # if lwrist_pose is not None:
        #     pose_dict["smplx_body_pose"][57:60] = lwrist_pose
        # if rwrist_pose is not None:
        #     pose_dict["smplx_body_pose"][60:63] = rwrist_pose
        if lwrist_pose is not None and rwrist_pose is not None:
            ava_list.append(i)

        rend_data.append(pose_dict)


vid.release()

print(ava_list)

# from animation import *
#
# play_pose_parameters(rend_data, output_dir="vid_ori")