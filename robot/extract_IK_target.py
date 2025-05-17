import os.path

import mujoco
import mujoco.viewer
import numpy as np

from scipy.spatial.transform import Rotation as R

# Load your robot
model = mujoco.MjModel.from_xml_path(os.path.join("h1_2", "scene.xml"))
data = mujoco.MjData(model)

from math import pi

import pickle

with open(os.path.join("../pose_data", "1.pkl"), "rb") as f:
    smplx_data = pickle.load(f)

def smpl2h1RotMatrix(smplx_rotation_matrix):
    # h1->smplx
    axis_trans = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]
    h1_rotation_matrix = np.matmul(np.matmul(np.transpose(axis_trans), smplx_rotation_matrix), axis_trans)
    return h1_rotation_matrix

# I have implemented the retarget pipeline for the upper body, but there is some mathmatic error, I am trying to derive the whole process rigorously to fix the error.

# retarget the default pose
R1_left_shoulder_matrix = R.from_euler(seq="x", angles=90, degrees=True).as_matrix()
R1_right_shoulder_matrix = R.from_euler(seq="x", angles=-90, degrees=True).as_matrix()

R1_left_elbow_matrix = np.matmul(R1_left_shoulder_matrix, R.from_euler(seq="y", angles=90, degrees=True).as_matrix())
R1_right_elbow_matrix = np.matmul(R1_right_shoulder_matrix, R.from_euler(seq="y", angles=90, degrees=True).as_matrix())

# R1_left_elbow_matrix = R.from_euler(seq="y", angles=90, degrees=True).as_matrix()
# R1_right_elbow_matrix = R.from_euler(seq="y", angles=90, degrees=True).as_matrix()

def smplAxisAngle2H1RotMatrix(smplx_rotvec_list:list, body_part, is_right):
    if body_part == "shoulder" or body_part == "shoudler":
        if is_right:
            R_default_matrix = R1_right_shoulder_matrix
        else:
            R_default_matrix = R1_left_shoulder_matrix
    elif body_part == "elbow":
        if is_right:
            R_default_matrix = R1_right_elbow_matrix
        else:
            R_default_matrix = R1_left_elbow_matrix
    else:
        if is_right:
            R_default_matrix = R1_right_elbow_matrix
        else:
            R_default_matrix = R1_left_elbow_matrix
    R_smplx_matrix = np.eye(3)
    for smplx_rotvec in smplx_rotvec_list:
        R_smplx_joint_matrix = R.from_rotvec(smplx_rotvec, degrees=False).as_matrix()
        R_smplx_matrix = R_smplx_matrix @ R_smplx_joint_matrix
    R_h1_matrix = smpl2h1RotMatrix(R_smplx_matrix)
    return R_h1_matrix, R_default_matrix, R_smplx_matrix


dof_names = []
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    dof_names.append(joint_name)

body_names = []
for bid in range(model.nbody):                                 # 0 … nbody-1
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    body_names.append(body_name)

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
rest_xpos = {}
for jid in range(model.njnt):
    bid = model.jnt_bodyid[jid]
    rest_xpos[dof_names[jid]] = data.xpos[bid]

root2left_shoudler_vec = rest_xpos["left_shoulder_pitch_joint"] - rest_xpos["floating_base_joint"]
root2right_shoudler_vec = rest_xpos["right_shoulder_pitch_joint"] - rest_xpos["floating_base_joint"]

left_shoudler2elbow_vec = rest_xpos["left_elbow_pitch_joint"] - rest_xpos["left_shoulder_pitch_joint"]
right_shoudler2elbow_vec = rest_xpos["right_elbow_pitch_joint"] - rest_xpos["right_shoulder_pitch_joint"]

left_elbow2wrist_vec = rest_xpos["left_wrist_pitch_joint"] - rest_xpos["left_elbow_pitch_joint"]
right_elbow2wrist_vec = rest_xpos["right_wrist_pitch_joint"] - rest_xpos["right_elbow_pitch_joint"]

nframes = smplx_data['smplx_body_pose'].shape[0]
IK_target_list = []
for frame in range(nframes):
    IK_target_dic = {}

    left_collar_axis_angle = smplx_data["smplx_body_pose"][frame][12*3:12*3+3]
    right_collar_axis_angle = smplx_data["smplx_body_pose"][frame][13*3:13*3+3]

    left_shoulder_axis_angle = smplx_data["smplx_body_pose"][frame][15*3:15*3+3]
    right_shoulder_axis_angle = smplx_data["smplx_body_pose"][frame][16*3:16*3+3]

    # shoulder
    left_shoulder_matrix_h1, left_shoulder_matrix_default, left_shoulder_matrix_smplx = smplAxisAngle2H1RotMatrix([left_collar_axis_angle, left_shoulder_axis_angle], body_part="shoulder", is_right=False)
    right_shoulder_matrix_h1, right_shoulder_matrix_default, _ = smplAxisAngle2H1RotMatrix([right_collar_axis_angle, right_shoulder_axis_angle], body_part="shoulder", is_right=True)

    # elbow
    left_elbow_axis_angle = smplx_data["smplx_body_pose"][frame][17*3:17*3+3]
    right_elbow_axis_angle = smplx_data["smplx_body_pose"][frame][18*3:18*3+3]

    left_elbow_matrix_h1, left_elbow_matrix_default, left_elbow_matrix_smplx = smplAxisAngle2H1RotMatrix([left_elbow_axis_angle], body_part="elbow", is_right=False)
    right_elbow_matrix_h1, right_elbow_matrix_default, _ = smplAxisAngle2H1RotMatrix([right_elbow_axis_angle], body_part="elbow", is_right=True)

    # wrist
    left_wrist_axis_angle = smplx_data["smplx_body_pose"][frame][19*3:19*3+3]
    right_wrist_axis_angle = smplx_data["smplx_body_pose"][frame][20*3:20*3+3]

    left_wrist_matrix_h1, left_wrist_matrix_default, left_wrist_matrix_smplx = smplAxisAngle2H1RotMatrix([left_wrist_axis_angle], body_part="wrist", is_right=False)
    right_wrist_matrix_h1, right_wrist_matrix_default, _ = smplAxisAngle2H1RotMatrix([right_wrist_axis_angle], body_part="wrist", is_right=True)

    left_wrist_pos = root2left_shoudler_vec + left_shoulder_matrix_h1 @ left_shoulder_matrix_default @ left_shoudler2elbow_vec + left_shoulder_matrix_h1 @ left_elbow_matrix_h1 @ left_elbow_matrix_default @ left_elbow2wrist_vec
    right_wrist_pos = root2right_shoudler_vec + right_shoulder_matrix_h1 @ right_shoulder_matrix_default @ right_shoudler2elbow_vec + right_shoulder_matrix_h1 @ right_elbow_matrix_h1 @ right_elbow_matrix_default @ right_elbow2wrist_vec

    IK_target_dic["left_wrist_pos"] = left_wrist_pos
    IK_target_dic["right_wrist_pos"] = right_wrist_pos

    left_wrist_matrix_h1_world = left_shoulder_matrix_h1 @ left_elbow_matrix_h1 @ left_wrist_matrix_h1 @ left_wrist_matrix_default
    right_wrist_matrix_h1_world = right_shoulder_matrix_h1 @ right_elbow_matrix_h1 @ right_wrist_matrix_h1 @ right_wrist_matrix_default

    # forward_direc = left_wrist_matrix_h1_world @ np.array([1, 0, 0])

    left_wrist_matrix_h1_world_from_smplx = smpl2h1RotMatrix(left_shoulder_matrix_smplx @ left_elbow_matrix_smplx @ left_wrist_matrix_smplx)

    # forward_direc2 = left_shoulder_matrix_smplx @ left_elbow_matrix_smplx @ left_wrist_matrix_smplx @ np.array([1, 0, 0])

    left_wrist_matrix_h1_rel = np.transpose(left_wrist_matrix_default) @ left_wrist_matrix_h1 @ left_wrist_matrix_default
    right_wrist_matrix_h1_rel = np.transpose(right_wrist_matrix_default) @ right_wrist_matrix_h1 @ right_wrist_matrix_default

    IK_target_dic["left_wrist_euler"] = R.from_matrix(left_wrist_matrix_h1_world).as_euler(seq="xyz", degrees=False)
    IK_target_dic["right_wrist_euler"] = R.from_matrix(right_wrist_matrix_h1_world).as_euler(seq="xyz", degrees=False)

    IK_target_list.append(IK_target_dic)

    left_shoulder_euler_degree = R.from_matrix(left_shoulder_matrix_smplx).as_euler(seq="xyz", degrees=True)
    left_elbow_euler_degree = R.from_matrix(left_elbow_matrix_smplx).as_euler(seq="xyz", degrees=True)
    left_wrist_euler_degree = R.from_matrix(left_wrist_matrix_smplx).as_euler(seq="xyz", degrees=True)

with open("IK_target_list.pkl", "wb") as f:
    pickle.dump(IK_target_list, f)



