import os.path

import casadi
import math
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer

np.set_printoptions(precision=5, suppress=True, linewidth=200)

urdf_path = os.path.join('h1_2', 'h1_2.urdf')
model_dir = os.path.dirname(urdf_path)
mesh_dir = os.path.join(model_dir, "meshes")

robot = pin.RobotWrapper.BuildFromURDF(urdf_path, [model_dir, mesh_dir])
joints_to_lock = ["left_hip_yaw_joint",
                  "left_hip_pitch_joint",
                  "left_hip_roll_joint",
                  "left_knee_joint",
                  "left_ankle_pitch_joint",
                  "left_ankle_roll_joint",
                  "right_hip_yaw_joint",
                  "right_hip_pitch_joint",
                  "right_hip_roll_joint",
                  "right_knee_joint",
                  "right_ankle_pitch_joint",
                  "right_ankle_roll_joint",
                  "torso_joint",
                  "L_index_proximal_joint",
                  "L_index_intermediate_joint",
                  "L_middle_proximal_joint",
                  "L_middle_intermediate_joint",
                  "L_pinky_proximal_joint",
                  "L_pinky_intermediate_joint",
                  "L_ring_proximal_joint",
                  "L_ring_intermediate_joint",
                  "L_thumb_proximal_yaw_joint",
                  "L_thumb_proximal_pitch_joint",
                  "L_thumb_intermediate_joint",
                  "L_thumb_distal_joint",
                  "R_index_proximal_joint",
                  "R_index_intermediate_joint",
                  "R_middle_proximal_joint",
                  "R_middle_intermediate_joint",
                  "R_pinky_proximal_joint",
                  "R_pinky_intermediate_joint",
                  "R_ring_proximal_joint",
                  "R_ring_intermediate_joint",
                  "R_thumb_proximal_yaw_joint",
                  "R_thumb_proximal_pitch_joint",
                  "R_thumb_intermediate_joint",
                  "R_thumb_distal_joint"]
h1_dual_arm = robot.buildReducedRobot(
    list_of_joints_to_lock=joints_to_lock,
    reference_configuration=np.array([0.0] * robot.model.nq),
)

# Create end effector in hand
h1_dual_arm.model.addFrame(
    pin.Frame('L_ee',
              h1_dual_arm.model.getJointId('left_wrist_yaw_joint'),
              pin.SE3(np.eye(3),
                      np.array([0.15, 0, 0]).T),
              pin.FrameType.OP_FRAME))
h1_dual_arm.model.addFrame(
    pin.Frame('R_ee',
              h1_dual_arm.model.getJointId('right_wrist_yaw_joint'),
              pin.SE3(np.eye(3),
                      np.array([0.15, 0, 0]).T),
              pin.FrameType.OP_FRAME))

h1_dual_arm.setVisualizer(MeshcatVisualizer())
h1_dual_arm.initViewer(open=True)
h1_dual_arm.loadViewerModel("pinocchio")
# Enable the display of end effector frames
h1_dual_arm.viz.displayFrames(
    True,
    frame_ids=[h1_dual_arm.model.getFrameId('L_ee'),
               h1_dual_arm.model.getFrameId('R_ee')],
    axis_length=0.2,
    axis_width=5)
h1_dual_arm.display(pin.neutral(h1_dual_arm.model))

frame_viz_names = ['L_ee_target', 'R_ee_target']
FRAME_AXIS_POSITIONS = (
    np.array([[0, 0, 0], [1, 0, 0],
              [0, 0, 0], [0, 1, 0],
              [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
)
FRAME_AXIS_COLORS = (
    np.array([[1, 0, 0], [1, 0.6, 0],
              [0, 1, 0], [0.6, 1, 0],
              [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
)
axis_length = 0.1
axis_width = 10
for frame_viz_name in frame_viz_names:
    h1_dual_arm.viewer[frame_viz_name].set_object(
        mg.LineSegments(
            mg.PointsGeometry(
                position=axis_length * FRAME_AXIS_POSITIONS,
                color=FRAME_AXIS_COLORS,
            ),
            mg.LineBasicMaterial(
                linewidth=axis_width,
                vertexColors=True,
            ),
        )
    )


cmodel = cpin.Model(h1_dual_arm.model)
cdata = cmodel.createData()
cq = casadi.SX.sym("q", h1_dual_arm.model.nq, 1)

# Compute kinematics casadi graphs
cpin.framesForwardKinematics(cmodel, cdata, cq)

cL_ee_target = casadi.SX.sym("L_ee_target", 4, 4)
cR_ee_target = casadi.SX.sym("R_ee_target", 4, 4)
L_hand_id = h1_dual_arm.model.getFrameId("L_ee")
R_hand_id = h1_dual_arm.model.getFrameId("R_ee")
error_func = casadi.Function(
    "error_func",
    [cq, cL_ee_target, cR_ee_target],
    [
        casadi.vertcat(
            cpin.log6(
                cdata.oMf[L_hand_id].inverse() * cpin.SE3(cL_ee_target)
            ).vector,
            cpin.log6(
                cdata.oMf[R_hand_id].inverse() * cpin.SE3(cR_ee_target)
            ).vector
        )
    ],
)

opti = casadi.Opti()
# Create variables and parameters
var_q = opti.variable(h1_dual_arm.model.nq)  # 14 DOF joints of dual arm
param_q_ik_last = opti.parameter(h1_dual_arm.model.nq)
param_tf_l = opti.parameter(4, 4)
param_tf_r = opti.parameter(4, 4)

# Create costs
ee_cost = casadi.sumsqr(error_func(var_q, param_tf_l, param_tf_r))
regularization_cost = casadi.sumsqr(var_q)
smooth_cost = casadi.sumsqr(var_q - param_q_ik_last)
w_ee = 20.0
w_sm = 0.1
w_re = 0.001
opti.minimize(w_ee * ee_cost +
              w_sm * smooth_cost +
              w_re * regularization_cost)

# Create constraints
opti.subject_to(opti.bounded(
    h1_dual_arm.model.lowerPositionLimit,
    var_q,
    h1_dual_arm.model.upperPositionLimit)
)

# Using the IPopt
plugin_options = {'print_time': False}
ipopt_options = {'tol': 1e-3, 'print_level': 1}
opti.solver('ipopt', plugin_options, ipopt_options)

def set_target_pose(x, y, z, roll, pitch, yaw):
    """Returns an SE3 transformation for given x, y, z, roll, pitch, yaw."""
    rotation = pin.utils.rpyToMatrix(roll, pitch, yaw)
    translation = np.array([x, y, z])
    return pin.SE3(rotation, translation)

# Example input values for target positions and orientations
left_target_pose = {'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0}
right_target_pose = {'x': 40, 'y': 0, 'z': 3, 'roll': 0, 'pitch': 0, 'yaw': np.pi / 8}

q_warm = np.zeros(h1_dual_arm.model.nq)

# Set the target poses based on provided x, y, z, roll, pitch, yaw
L_target = set_target_pose(**left_target_pose)
R_target = set_target_pose(**right_target_pose)

# Set initial and target values for optimization
opti.set_initial(var_q, q_warm)
opti.set_value(param_q_ik_last, q_warm)
opti.set_value(param_tf_l, L_target.homogeneous)
opti.set_value(param_tf_r, R_target.homogeneous)

# Show targets
h1_dual_arm.viewer['L_ee_target'].set_transform(L_target.homogeneous)
h1_dual_arm.viewer['R_ee_target'].set_transform(R_target.homogeneous)

try:
    opti.solve()
    q_result = opti.value(var_q)
    h1_dual_arm.display(q_result)
    q_warm = q_result
    print("Success in Convergence!")
    print("Press <Enter> to quit …")
    input()
except:
    print("Failed in Convergence!")
    q_result = opti.debug.value(var_q)
    h1_dual_arm.display(q_result)
    q_warm = np.zeros(h1_dual_arm.model.nq)  # Reset warm start