import json
import math
import os.path

from smplx import SMPLX
from smplx import MANOLayer, create
import torch
from dataset_util import transfer_phonenix2render

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QSpinBox, \
    QMessageBox, QComboBox, QLabel

import sys
import cv2
import numpy as np

from scipy.spatial.transform import Rotation as R

import pickle
import pandas as pd
mano_pickle = pd.read_pickle('data/mano/MANO_RIGHT.pkl')
hands_mean_right = pd.read_pickle('data/mano/MANO_RIGHT.pkl')['hands_mean']
hands_mean_left  = pd.read_pickle('data/mano/MANO_LEFT.pkl')['hands_mean']

class Demo(QWidget):
    def __init__(self, index, show_keypoints = False, mesh_type="smplx"):
        super().__init__()
        self.mesh_type = mesh_type
        self.batch_size = 224
        self.show_keypoints = show_keypoints

        self.window = gl.GLViewWidget()
        self.window.setGeometry(1000, 0, 1920, 1080)
        self.window.setCameraPosition(distance=1.5, elevation=1.8)
        self.window.setBackgroundColor(0.8)
        self.window.show()

        zgrid = gl.GLGridItem(color=(255, 255, 255, 226))
        self.window.addItem(zgrid)

        device = torch.device('cpu')
        if mesh_type == "smplx":
            smplx = SMPLX(device=device, model_path=os.path.join("..", "smpl_model", "smplx", "SMPLX_FEMALE.npz"), use_pca=False, flat_hand_mean=False)
            # color different part
            with open(os.path.join("..", "smpl_model", "smplx_vert_segmentation.json")) as f:
                smplx_seg = json.load(f)
                vertex2part_map = {}
                count = 0
                for key, value in smplx_seg.items():
                    for vertex in value:
                        count += 1
                        if vertex2part_map.get(vertex) is not None:
                            vertex2part_map[vertex].append(key)
                        vertex2part_map[vertex] = [key]
                for i in range(10469):
                    if vertex2part_map.get(i) is None:
                        print(i)
                import matplotlib.pyplot as plt
                cmap = plt.get_cmap("plasma")
                keys = list(smplx_seg.keys())
                color_array = [cmap(i/len(keys)) for i in range(len(keys))]
                color_dict = {}
                for i, key in enumerate(keys):
                    color_dict[key] = color_array[i]
                self.vertex_color_array = np.zeros((10475, 4))
                for key, values in vertex2part_map.items():
                    color = np.zeros(4)
                    count = 0
                    for value in values:
                        color += color_dict[value]
                        count += 1
                    color = color / count
                    self.vertex_color_array[key] = color

            self.faces = smplx.faces.astype(int)

            smplx_betas = []
            root_poses = []
            body_poses = []
            lhand_poses = []
            rhand_poses = []

            if self.show_keypoints:
                key_name, pose_list = transfer_phonenix2render(file_name, index = index)
                with open(keypoints_file_name, "rb") as f:
                    self.keypoints = pickle.load(f)[key_name]["keypoints"]
            # keypoints_list =
            with open(pose_file, "rb") as f:
                pose_list = pickle.load(f)
            for pose in pose_list:
                root_poses.append(pose['smplx_root_pose'])
                smplx_betas.append([0]*10)
                body_poses.append(pose["smplx_body_pose"])
                lhand_poses.append(pose["smplx_lhand_pose"])
                rhand_poses.append(pose["smplx_rhand_pose"])
            # for file_name in file_name_list:
            #     directory = os.path.join(dir, file_name)
            #     with open(directory, 'rb') as f:
            #         pickle_data = pickle.load(f)
            #         root_poses.append(pickle_data['smplx_root_pose'])
            #         smplx_betas.append(pickle_data["smplx_shape"])
            #         body_poses.append(pickle_data["smplx_body_pose"])
            #         lhand_poses.append(pickle_data["smplx_lhand_pose"])
            #         rhand_poses.append(pickle_data["smplx_rhand_pose"])


            batch_size = len(pose_list)

            smplx_betas = np.array(smplx_betas)
            smplx_betas = torch.from_numpy(smplx_betas).float()
            body_poses = np.array(body_poses)
            body_poses = torch.from_numpy(body_poses).float().view(-1, 21, 3)
            root_poses = np.array(root_poses)
            root_poses = torch.from_numpy(root_poses).float().view(-1, 1, 3)
            lhand_poses = np.array(lhand_poses)
            lhand_poses = torch.from_numpy(lhand_poses).float()
            rhand_poses = np.array(rhand_poses)
            rhand_poses = torch.from_numpy(rhand_poses).float()

            leye_pose = torch.zeros((batch_size, 1, 3))
            reye_pose = torch.zeros((batch_size, 1, 3))
            jaw_pose = torch.zeros((batch_size, 1, 3))
            expressions = torch.zeros((batch_size, 10))
            outputs = smplx.forward(expression=expressions, jaw_pose=jaw_pose, leye_pose= leye_pose, reye_pose= reye_pose, global_orient=root_poses, betas=smplx_betas, body_pose=body_poses, left_hand_pose=lhand_poses, right_hand_pose=rhand_poses)
            self.vertices = outputs["vertices"].detach().cpu().numpy()

            self.pointcolors = (100, 100, 100)

            # self.joints = np.load("pose.npy")
            # self.points_item = gl.GLScatterPlotItem(pos=self.joints[0], color=pg.glColor(self.pointcolors))
        elif mesh_type == "mano":
            mano_right = create(
                "data",
                model_type='mano',
                is_rhand=True,
                use_pca=False
            )
            mano_right_faces = mano_right.faces
            mano_left = create(
                "data",
                model_type='mano',
                is_rhand=False,
                use_pca=False
            )
            mano_left_faces = mano_left.faces
            self.vertices = []
            with open(file_name, "rb") as f:
                hand_poses = pickle.load(f)
            if hand_poses is not None:
                for i in range(len(hand_poses)):
                    if hand_poses[i].get("smplx_rhand_pose") is not None:
                        right_hand_poses = np.array(hand_poses[i]["smplx_rhand_pose"]) - hands_mean_right
                        wrist_pose = torch.from_numpy(np.array(hand_poses[i]["rwrist_global_orient"]).reshape(-1, 3)).type(torch.FloatTensor)
                        # wrist_pose = torch.zeros((1, 3))
                        # Convert wrist_pose from axis-angle to rotation matrix
                        wrist_pose_matrix = R.from_rotvec(wrist_pose, degrees=False).as_matrix()

                        rotation_matrix_x = torch.tensor([
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]
                        ], dtype=torch.float32)

                        # Apply the rotation around the x-axis
                        rotated_res = torch.matmul(rotation_matrix_x, torch.from_numpy(wrist_pose_matrix).float())
                        wrist_pose = torch.from_numpy(R.from_matrix(rotated_res).as_rotvec(degrees=False)).type(torch.FloatTensor)


                        right_hand_poses = torch.from_numpy(right_hand_poses.reshape(-1, 45)).type(torch.FloatTensor)
                        betas = torch.from_numpy(np.zeros((right_hand_poses.shape[0], 10))).type(torch.FloatTensor)
                        output = mano_right.forward(hand_pose=right_hand_poses, betas=betas, global_orient=wrist_pose)
                        vertices = output["vertices"].detach().cpu().numpy()[0]
                        self.vertices.append(vertices)
                    # if hand_poses[i].get("smplx_lhand_pose") is not None:
                    #     left_hand_poses = np.array(hand_poses[i]["smplx_lhand_pose"]) - hands_mean_right
                    #     wrist_pose = torch.from_numpy(np.array(hand_poses[i]["lwrist_global_orient"]).reshape(-1, 3)).type(torch.FloatTensor)
                    #     # wrist_pose = torch.zeros((1, 3))
                    #
                    #     wrist_pose_matrix = R.from_rotvec(wrist_pose, degrees=False).as_matrix()
                    #
                    #     rotation_matrix_x = torch.tensor([
                    #         [1, 0, 0],
                    #         [0, 0, 1],
                    #         [0, -1, 0]
                    #     ], dtype=torch.float32)
                    #
                    #     # Apply the rotation around the x-axis
                    #     rotated_res = torch.matmul(rotation_matrix_x, torch.from_numpy(wrist_pose_matrix).float())
                    #     wrist_pose = torch.from_numpy(R.from_matrix(rotated_res).as_rotvec(degrees=False)).type(torch.FloatTensor)
                    #
                    #     left_hand_poses = torch.from_numpy(left_hand_poses.reshape(-1, 45)).type(torch.FloatTensor)
                    #     betas = torch.from_numpy(np.zeros((left_hand_poses.shape[0], 10))).type(torch.FloatTensor)
                    #     output = mano_left.forward(hand_pose=left_hand_poses, betas=betas, global_orient=wrist_pose)
                    #     vertices = output["vertices"].detach().cpu().numpy()[0]
                    #     self.vertices.append(vertices)
                    # self.vertices.append(np.array(hand_poses[i]["rhand_pred_vertices"]).reshape(-1, 3))
                self.vertices = np.array(self.vertices)
                # self.faces = mano_right_faces
                self.faces = mano_left_faces

        if mesh_type == "smplx":
            meshdata = gl.MeshData(vertexes=self.vertices[0], faces=self.faces, vertexColors=self.vertex_color_array)
        elif mesh_type == "mano":
            meshdata = gl.MeshData(vertexes=self.vertices[0], faces=self.faces)
        self.mesh_item = gl.GLMeshItem(meshdata=meshdata)

        if self.show_keypoints :
            self.scatter_item = gl.GLScatterPlotItem(pos = self.keypoints[0], color=(1, 0, 0, 1))
            self.window.addItem(self.scatter_item)

        self.N_frames = self.vertices.shape[0]
        self.window.addItem(self.mesh_item)
        self.frame_number = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(150)
        self.timer.start()

        # Add coordinate axes
        # Create lines for x, y, z axes
        axis_length = 1.0  # Length of axis lines
        
        # X-axis (red)
        x_axis = np.array([[0, 0, 0], [axis_length, 0, 0]])
        x_line = gl.GLLinePlotItem(pos=x_axis, color=(1, 0, 0, 1), width=5)
        self.window.addItem(x_line)
        
        # Y-axis (green)
        y_axis = np.array([[0, 0, 0], [0, axis_length, 0]])
        y_line = gl.GLLinePlotItem(pos=y_axis, color=(0, 1, 0, 1), width=5)
        self.window.addItem(y_line)
        
        # Z-axis (blue)
        z_axis = np.array([[0, 0, 0], [0, 0, axis_length]])
        z_line = gl.GLLinePlotItem(pos=z_axis, color=(0, 0, 1, 1), width=5)
        self.window.addItem(z_line)
        
        # Add labels for axes
        x_label = gl.GLTextItem(pos=[axis_length, 0, 0], text='X', color=(1, 0, 0, 1))
        y_label = gl.GLTextItem(pos=[0, axis_length, 0], text='Y', color=(0, 1, 0, 1))
        z_label = gl.GLTextItem(pos=[0, 0, axis_length], text='Z', color=(0, 0, 1, 1))
        
        self.window.addItem(x_label)
        self.window.addItem(y_label)
        self.window.addItem(z_label)

    def update_frame(self):
        self.frame_number += 1
        if self.frame_number >= self.N_frames:
            self.frame_number =0
        self.frame_number = 0
        # self.points_item.setData(pos=self.joints[self.frame_number])
        if self.mesh_type == "smplx":
            self.mesh_item.setMeshData(vertexes=self.vertices[self.frame_number], faces=self.faces, vertexColors=self.vertex_color_array)
        elif self.mesh_type == "mano":
            self.mesh_item.setMeshData(vertexes=self.vertices[self.frame_number], faces=self.faces)
        if self.show_keypoints:
            self.scatter_item.setData(pos= self.keypoints[self.frame_number])
        self.window.update()


keypoints_file_name = os.path.join("..", "SLRT", "Spoken2Sign", "data", "phoenix-2014t-keypoints.pkl")
if __name__ == '__main__':
    # file_name = os.path.join("data", "phoenix_000000_001500.pkl")
    # file_name = os.path.join("pose_data", "_2FBDaOPYig_1-3-rgb_front" + "_hand_refine.pkl")
    file_name = os.path.join("pose_data", "output_video" + "_hand_refine.pkl")
    pose_file = "combined_pose.pkl"
    # file_name_list = os.listdir(dir)
    app = QApplication(sys.argv)
    t = Demo(index=0,  show_keypoints = False, mesh_type="mano")
    sys.exit(app.exec_())