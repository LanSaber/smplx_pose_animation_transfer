import os.path

import mujoco
import mujoco.viewer
import numpy as np

from scipy.spatial.transform import Rotation as R

# Load your robot
model = mujoco.MjModel.from_xml_path(os.path.join("h1_2", "scene.xml"))
data = mujoco.MjData(model)

from math import pi

default_pose = {
    # Left arm - slightly forward for balance
    'left_shoulder_pitch_joint': 0,
    'left_shoulder_roll_joint': pi / 2,
    'left_shoulder_yaw_joint': 0,

    # 'left_elbow_pitch_joint': pi / 2,
    # 'left_elbow_roll_joint': 0,
    # #
    # 'left_wrist_pitch_joint': 0,
    # 'left_wrist_yaw_joint': 0,

    # Right arm - slightly forward for balance
    'right_shoulder_pitch_joint': 0,
    'right_shoulder_roll_joint': -pi / 2,
    'right_shoulder_yaw_joint': 0,

    # 'right_elbow_pitch_joint': pi / 2,
    # 'right_elbow_roll_joint': 0,
    # #
    # 'right_wrist_pitch_joint': 0,
    # 'right_wrist_yaw_joint': 0,
}

import pickle
with open(os.path.join("../pose_data", "1.pkl"), "rb") as f:
    smplx_data = pickle.load(f)


fps = 20
frame_num = smplx_data["smplx_body_pose"].shape[0]

dof_names = []
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    dof_names.append(joint_name)

body_names = []
for bid in range(model.nbody):                                 # 0 … nbody-1
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    body_names.append(body_name)

print("Degrees of Freedom (DOF) names:", dof_names)


target_qpos = np.copy(data.qpos)  # Target is initial pose (standing)
for name, target_angle in default_pose.items():
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qpos_idx = model.jnt_qposadr[joint_id]
    data.qpos[qpos_idx] = target_angle


# h1->smplx
axis_trans = [
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
]

def apply_animation(model, data, time):
    previous_frame = int(time // (1000 / fps)) % frame_num
    next_frame = (previous_frame + 1) % frame_num
    previous_body = smplx_data["smplx_body_pose"][previous_frame]
    next_body = smplx_data["smplx_body_pose"][next_frame]

    previous_body = smplx_data["smplx_body_pose"][0]
    next_body = smplx_data["smplx_body_pose"][0]

    # Calculate the interpolation factor based on the current time
    interpolation_factor = (time % (1000 / fps)) / (1000 / fps)

    # Perform spherical linear interpolation (slerp) between the previous and next body poses
    interpolated_body = (1 - interpolation_factor) * previous_body + interpolation_factor * next_body

    # Shoulder map
    R1_left_shoulder = R.from_euler("x", 90, degrees=True).as_matrix()
    R1_right_shoulder = R.from_euler("x", -90, degrees=True).as_matrix()

    left_shoudler_axis_angle = interpolated_body[15*3:15*3+3]
    right_shoudler_axis_angle = interpolated_body[16*3:16*3+3]

    left_shoulder_rotation_smplx = R.from_rotvec(left_shoudler_axis_angle)
    right_shoulder_rotation_smplx = R.from_rotvec(right_shoudler_axis_angle)

    left_shoulder_matrix_smplx = left_shoulder_rotation_smplx.as_matrix()
    right_shoulder_matrix_smplx = right_shoulder_rotation_smplx.as_matrix()

    left_shoulder_euler_smplx = left_shoulder_rotation_smplx.as_euler('ZYX', degrees=True)
    right_shoulder_euler_smplx = right_shoulder_rotation_smplx.as_euler('ZYX', degrees=True)

    R2_left_shoulder = np.matmul(np.matmul(np.linalg.inv(axis_trans), left_shoulder_matrix_smplx), axis_trans)
    R2_right_shoulder = np.matmul(np.matmul(np.linalg.inv(axis_trans), right_shoulder_matrix_smplx), axis_trans)

    left_shoulder_euler2_smplx = R.from_matrix(R2_left_shoulder).as_euler('XYZ', degrees=True)
    right_shoulder_euler2_smplx = R.from_matrix(R2_right_shoulder).as_euler('XYZ', degrees=True)

    left_shoulder_matrix = np.matmul(R2_left_shoulder, R1_left_shoulder)
    right_shoulder_matrix = np.matmul(R2_right_shoulder, R1_right_shoulder)


    left_shoudler_euler = R.from_matrix(left_shoulder_matrix).as_euler(seq="YXZ", degrees=False)
    # left_shoudler_euler[[0, 1, 2]] = left_shoudler_euler[[0, 2, 1]]
    right_shoudler_euler = R.from_matrix(right_shoulder_matrix).as_euler(seq="YXZ", degrees=False)
    # right_shoudler_euler[[0, 1, 2]] = right_shoudler_euler[[0, 2, 1]]

    # Elbow map

    R1_left_elbow = R.from_euler("y", 90, degrees=True).as_matrix()
    R1_right_elbow = R.from_euler("y", 90, degrees=True).as_matrix()


    R1_left_elbow = np.matmul(R1_left_shoulder, R1_left_elbow)
    R1_right_elbow = np.matmul(R1_right_shoulder, R1_right_elbow)

    left_elbow_axis_angle = interpolated_body[17*3:17*3+3]

    angle_radius = np.linalg.norm(left_elbow_axis_angle, axis=0)
    angle_degree = angle_radius / pi * 180
    axis = left_elbow_axis_angle / angle_radius

    right_elbow_axis_angle = interpolated_body[18*3:18*3+3]

    left_elbow_rotation_smplx = R.from_rotvec(left_elbow_axis_angle, degrees=False)

    left_elbow_matrix_smplx = left_elbow_rotation_smplx.as_matrix()


    right_elbow_rotation_smplx = R.from_rotvec(right_elbow_axis_angle, degrees=False)

    left_elbow_euler_smplx = left_elbow_rotation_smplx.as_euler(seq="xyz", degrees=False)

    right_elbow_euler_smplx = right_elbow_rotation_smplx.as_euler(seq="xyz", degrees=True)

    left_elbow_matrix_smplx = left_elbow_rotation_smplx.as_matrix()
    right_elbow_matrix_smplx = right_elbow_rotation_smplx.as_matrix()

    R2_left_elbow = np.matmul(np.matmul(np.linalg.inv(axis_trans), left_elbow_matrix_smplx), axis_trans)
    R2_right_elbow = np.matmul(np.matmul(np.linalg.inv(axis_trans), right_elbow_matrix_smplx), axis_trans)


    left_elbow_matrix = np.matmul(R2_left_elbow, R1_left_elbow)
    right_elbow_matrix = np.matmul(R2_right_elbow, R1_right_elbow)

    left_elbow_euler = R.from_matrix(left_elbow_matrix).as_euler(seq="xyz", degrees=False)
    right_elbow_euler = R.from_matrix(right_elbow_matrix).as_euler(seq="xyz", degrees=False)

    # if left_elbow_euler[0]>

    # wrist map

    left_wrist_axis_angle = interpolated_body[19*3:19*3+3]
    right_wrist_axis_angle = interpolated_body[20*3:20*3+3]

    left_wrist_rotation_smplx = R.from_rotvec(left_wrist_axis_angle)
    right_wrist_rotation_smplx = R.from_rotvec(right_wrist_axis_angle)

    left_wrist_euler_smplx = left_wrist_rotation_smplx.as_euler(seq="XYZ", degrees=True)
    right_wrist_euler_smplx = right_wrist_rotation_smplx.as_euler(seq="XYZ", degrees=True)

    left_wrist_matrix_smplx = left_wrist_rotation_smplx.as_matrix()
    right_wrist_matrix_smplx = right_wrist_rotation_smplx.as_matrix()

    R2_left_wrist = np.matmul(np.matmul(np.linalg.inv(axis_trans), left_wrist_matrix_smplx), axis_trans)
    R2_right_wrist = np.matmul(np.matmul(np.linalg.inv(axis_trans), right_wrist_matrix_smplx), axis_trans)

    left_wrist_matrix = np.matmul(R2_left_wrist, R1_left_elbow)
    right_wrist_matrix = np.matmul(R2_right_wrist, R1_right_elbow)

    left_wrist_euler = R.from_matrix(left_wrist_matrix).as_euler(seq="xyz", degrees=True)
    right_wrist_euler = R.from_matrix(right_wrist_matrix).as_euler(seq="xyz", degrees=True)


    # robot_pose = np.concatenate((left_shoudler_euler, left_elbow_euler[0:2], left_wrist_euler[0::2], right_shoudler_euler, right_elbow_euler[0:2], right_wrist_euler[0::2]), axis=0)
    robot_pose = np.concatenate((left_shoudler_euler,
                                 right_shoudler_euler), axis=0)

    # Update the model's joint positions with the interpolated body pose
    for i, dof in enumerate(default_pose.keys()):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dof)
        qpos_idx = model.jnt_qposadr[joint_id]
        data.qpos[qpos_idx] = robot_pose[i]


# Calculate the time interval of each loop in the simulation
# The time interval can be determined by the simulation timestep and the number of substeps
# The timestep is defined in the model, and the number of substeps is typically 1 unless specified otherwise

# Get the simulation timestep
timestep = model.opt.timestep * 1000

# Calculate the time interval for each loop iteration
# Assuming the number of substeps is 1
time_interval = timestep

print("Time interval of each loop in the simulation:", time_interval)

with mujoco.viewer.launch_passive(model, data) as viewer:
    count = 0
    while viewer.is_running():
        apply_animation(model, data, count * timestep)
        mujoco.mj_forward(model, data)
        viewer.sync()
        count += 1
        for dof_name in dof_names:
            j_id = model.joint(dof_name).id  # numeric id
            anchor_xyz = data.xpos[j_id]  # (3,) vector
        body_pos = {}
        for body_name in body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            anchor_xyz = data.xpos[body_id]
            body_pos[body_name] = anchor_xyz
        dawe = 0
