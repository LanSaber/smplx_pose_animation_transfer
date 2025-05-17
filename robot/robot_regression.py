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

left_hand_regressor.load_state_dict(torch.load(os.path.join("..", "regressor", "left_hand_global_1_video_500_regressor.pth")))
right_hand_regressor.load_state_dict(torch.load(os.path.join("..", "regressor", "right_hand_global_1_video_500_regressor.pth")))

left_hand_regressor.eval()
right_hand_regressor.eval()

index = 502

input_video_path = os.path.join("..", "data", f"ca6e9ec9-46ac-4a02-8034-7c31157dc52c.mp4")

# Open input video
vid = cv2.VideoCapture(input_video_path)

total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float32)

ava_list = []

IK_target_data = []
last_lwrist_euler = np.zeros(3)
last_rwrist_euler = np.zeros(3)

last_lwrist_pos = np.zeros(3)
last_rwrist_pos = np.zeros(3)

# rotation_matrix_x = np.array([
#     [1, 0, 0],
#     [0, 0, 1],
#     [0, -1, 0]
# ],dtype=float)


for i in range(total_frames):
    vid.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, img_per_frame = vid.read()

    image = cv2.cvtColor(img_per_frame, cv2.COLOR_BGR2RGB)
    outputs = pipe.predict(image)

    lwrist_pose = None
    rwrist_pose = None


    lwrist_euler = last_lwrist_euler
    rwrist_euler = last_rwrist_euler

    lwrist_pos = last_lwrist_pos
    rwrist_pos = last_rwrist_pos

    if outputs.__len__()>2:
        continue

    for output in outputs:
        wilor_pred = output["wilor_preds"]
        hand_pose = torch.from_numpy(wilor_pred["hand_pose"].reshape(1, -1))
        cam_t = torch.from_numpy(wilor_pred['pred_cam_t_full'])
        global_orient = torch.from_numpy(wilor_pred['global_orient'][0])
        global_matrix = R.from_rotvec(global_orient.squeeze(0).cpu().detach().numpy(), degrees=False).as_matrix()
        input_tensor = torch.cat((global_orient, cam_t, hand_pose), dim=1)
        input_tensor = input_tensor.float().to(device)
        if output["is_right"] == 0.0:
            output_tensor = left_hand_regressor(input_tensor)
            output_tensor = output_tensor.detach().cpu()
            output_rotmat = torch.squeeze(rot6d_to_rotmat(output_tensor[:, 0:6])).numpy()

            # output_rotmat = rotation_matrix_init @ output_rotmat
            # output_rotmat = global_matrix
            lwrist_pose = R.from_matrix(output_rotmat).as_rotvec(degrees=False)
            lwrist_euler = R.from_matrix(output_rotmat).as_euler('xyz', degrees=False)
            lwrist_pos = torch.squeeze(output_tensor[:, 6:9]).detach().cpu().numpy()
        else:
            output_tensor = right_hand_regressor(input_tensor)
            output_tensor = output_tensor.detach().cpu()
            output_rotmat = torch.squeeze(rot6d_to_rotmat(output_tensor[:, 0:6])).numpy()

            # output_rotmat = rotation_matrix_init @ output_rotmat
            # output_rotmat = global_matrix
            rwrist_pose = R.from_matrix(output_rotmat).as_rotvec(degrees=False)
            rwrist_euler = R.from_matrix(output_rotmat).as_euler('xyz', degrees=False)
            rwrist_pos = torch.squeeze(output_tensor[:, 6:9]).detach().cpu().numpy()

    pose_dict = {}
    pose_dict["left_wrist_pos"] = lwrist_pos
    pose_dict["right_wrist_pos"] = rwrist_pos
    pose_dict["left_wrist_euler"] = lwrist_euler
    pose_dict["right_wrist_euler"] = rwrist_euler

    last_lwrist_pos = lwrist_pos
    last_rwrist_pos = rwrist_pos
    last_lwrist_euler = lwrist_euler
    last_rwrist_euler = rwrist_euler

    # if lwrist_pose is not None:
    #     pose_dict["smplx_body_pose"][57:60] = lwrist_pose
    # if rwrist_pose is not None:
    #     pose_dict["smplx_body_pose"][60:63] = rwrist_pose
    ava_list.append(i)

    IK_target_data.append(pose_dict)


vid.release()

print(ava_list)

with open('IK_list.pkl', 'wb') as f:
    pickle.dump(IK_target_data, f)