import torch
import cv2
import gzip
import os
import pickle
import wandb
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R

import argparse

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetRegression(nn.Module):
    def __init__(self, input_size=51, output_size=6):
        super(ResNetRegression, self).__init__()
        self.initial_fc = nn.Linear(input_size, 64)
        self.block1 = BasicBlock(64, 64)
        self.block2 = BasicBlock(64, 64)
        self.final_fc = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.initial_fc(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.final_fc(x)
        return x

def rotmat_to_rot6d(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to 6D rotation representation.
    
    Args:
        rotation_matrix: numpy array or torch tensor of shape (B, 3, 3)
        
    Returns:
        6D rotation representation as numpy array or torch tensor of shape (B, 6)
    """
    # Extract first two columns
    first_col = rotation_matrix[:, :, 0]
    second_col = rotation_matrix[:, :, 1]
    
    # Concatenate them
    rot6d = torch.concat((first_col, second_col), dim=1)
    
    return rot6d

def rot6d_to_rotmat(x):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    # Split the 6D vector into two 3D vectors
    a1 = x[:, :3]
    a2 = x[:, 3:]

    # Normalize the first vector
    b1 = F.normalize(a1)

    # Make the second vector orthogonal to the first
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)

    # Compute the third vector using cross product
    b3 = torch.cross(b1, b2)

    # Stack the vectors to form the rotation matrix
    rotation_matrix = torch.stack((b1, b2, b3), dim=-1)

    return rotation_matrix

def main():
    # Initialize wandb
    wandb.init(
        project="hand-pose-regression",
        name = f"global_{args.get('global')}_video_{args.get('video_nums')}",
        config={
            "learning_rate": 0.001,
            "architecture": "ResNetRegression",
            "dataset": "hand-pose",
            "epochs": "variable",
        }
    )

    left_hand_regressor = ResNetRegression().to(torch.device('cuda'))
    right_hand_regressor = ResNetRegression().to(torch.device('cuda'))
    loss_fn = torch.nn.MSELoss()
    left_optimizer = torch.optim.Adam(left_hand_regressor.parameters(), lr=0.001)
    right_optimizer = torch.optim.Adam(right_hand_regressor.parameters(), lr=0.001)

    with gzip.open(os.path.join("data", "labels_clean.all"), 'rb') as f:
        pose_dicts = pickle.load(f)
        keys = list(pose_dicts.keys())

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.float16

        pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)

        # Create output directories
        os.makedirs("visualization_mesh", exist_ok=True)
        os.makedirs("visualization_videos", exist_ok=True)

        for id in range(len(keys)):
            input_video_path = os.path.join("render_result", "vid_ego", f"{id}.mp4")

            # Open input video
            vid = cv2.VideoCapture(input_video_path)

            total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

            data = pose_dicts[keys[id]]["smpl"]

            for i in range(total_frames):
                # pelvis_matrix = R.from_rotvec(data["smplx_root_pose"][i]).as_matrix()
                pelvis_matrix = R.from_rotvec(data["smplx_body_pose"][i][0:3]).as_matrix()
                spine1_matrix = R.from_rotvec(data["smplx_body_pose"][i][6:9]).as_matrix()
                spine2_matrix = R.from_rotvec(data["smplx_body_pose"][i][15:18]).as_matrix()
                spine3_matrix = R.from_rotvec(data["smplx_body_pose"][i][24:27]).as_matrix()

                upper_body_matrix = np.matmul(spine1_matrix, np.matmul(spine2_matrix, spine3_matrix))
                # upper_body_matrix = np.identity(3)

                left_collar_matrix = R.from_rotvec(data["smplx_body_pose"][i][36:39]).as_matrix()
                right_collar_matrix = R.from_rotvec(data["smplx_body_pose"][i][39:42]).as_matrix()
                left_shoulder_matrix = R.from_rotvec(data["smplx_body_pose"][i][45:48]).as_matrix()
                right_shoulder_matrix = R.from_rotvec(data["smplx_body_pose"][i][48:51]).as_matrix()
                left_elbow_matrix = R.from_rotvec(data["smplx_body_pose"][i][51:54]).as_matrix()
                right_elbow_matrix = R.from_rotvec(data["smplx_body_pose"][i][54:57]).as_matrix()
                left_wrist_matrix_prefix = np.matmul(upper_body_matrix, np.matmul(left_collar_matrix,
                                                                                 np.matmul(left_shoulder_matrix,
                                                                                           left_elbow_matrix)))
                right_wrist_matrix_prefix = np.matmul(upper_body_matrix, np.matmul(right_collar_matrix,
                                                                                  np.matmul(right_shoulder_matrix,
                                                                                            right_elbow_matrix)))
                left_wrist_gt = R.from_rotvec(data["smplx_body_pose"][i][57:60]).as_matrix()
                right_wrist_gt = R.from_rotvec(data["smplx_body_pose"][i][60:63]).as_matrix()

                left_wrist_global_mat = np.matmul(left_wrist_matrix_prefix, left_wrist_gt)
                right_wrist_global_mat = np.matmul(right_wrist_matrix_prefix, right_wrist_gt)

                # global wrist orientation
                left_wrist_gt = R.from_matrix(left_wrist_global_mat).as_rotvec(degrees=False)
                right_wrist_gt = R.from_matrix(right_wrist_global_mat).as_rotvec(degrees=False)

                # relative wrist orientation
                left_wrist_gt_rel_mat = R.from_rotvec(data["smplx_body_pose"][i][57:60].reshape(1, -1)).as_matrix()
                right_wrist_gt_rel_mat = R.from_rotvec(data["smplx_body_pose"][i][60:63].reshape(1, -1)).as_matrix()

                left_wrist_gt = torch.from_numpy(left_wrist_gt.reshape(1, -1))
                right_wrist_gt = torch.from_numpy(right_wrist_gt.reshape(1, -1))

                if args.get("global") == 1:
                    # global 6d rotation
                    left_wrist_gt_6d = rotmat_to_rot6d(torch.from_numpy(left_wrist_global_mat.reshape(-1, 3, 3)))
                    right_wrist_gt_6d = rotmat_to_rot6d(torch.from_numpy(right_wrist_global_mat.reshape(-1, 3, 3)))
                else:
                    # relative 6d rotation
                    left_wrist_gt_6d = rotmat_to_rot6d(torch.from_numpy(left_wrist_gt_rel_mat.reshape(-1, 3, 3)))
                    right_wrist_gt_6d = rotmat_to_rot6d(torch.from_numpy(right_wrist_gt_rel_mat.reshape(-1, 3, 3)))


                left_wrist_gt_rotmat = rot6d_to_rotmat(left_wrist_gt_6d)
                right_wrist_gt_rotmat = rot6d_to_rotmat(right_wrist_gt_6d)

                vid.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, img_per_frame = vid.read()

                image = cv2.cvtColor(img_per_frame, cv2.COLOR_BGR2RGB)
                outputs = pipe.predict(image)

                for output in outputs:
                    wilor_pred = output["wilor_preds"]
                    hand_pose = torch.from_numpy(wilor_pred["hand_pose"].reshape(1, -1))
                    cam_t = torch.from_numpy(wilor_pred['pred_cam_t_full'])
                    global_orient = torch.from_numpy(wilor_pred['global_orient'][0])
                    input_tensor = torch.cat((global_orient, cam_t, hand_pose), dim=1)
                    input_tensor = input_tensor.float().to(device)

                    left_wrist_gt = left_wrist_gt.float().to(device)
                    right_wrist_gt = right_wrist_gt.float().to(device)

                    left_wrist_gt_6d = left_wrist_gt_6d.float().to(device)
                    right_wrist_gt_6d = right_wrist_gt_6d.float().to(device)
                    
                    if output["is_right"] == 0.0:
                        output_tensor = left_hand_regressor(input_tensor)
                        loss = loss_fn(output_tensor, left_wrist_gt_6d)
                        
                        # Log metrics for left hand
                        wandb.log({
                            "left_hand_loss": loss.item(),
                            "left_hand_learning_rate": left_optimizer.param_groups[0]['lr'],
                            "video_id": id
                        })

                        left_optimizer.zero_grad()
                        loss.backward()
                        left_optimizer.step()
                    else:
                        output_tensor = right_hand_regressor(input_tensor)
                        loss = loss_fn(output_tensor, right_wrist_gt_6d)
                        
                        # Log metrics for right hand
                        wandb.log({
                            "right_hand_loss": loss.item(),
                            "right_hand_learning_rate": right_optimizer.param_groups[0]['lr'],
                            "video_id": id
                        })

                        right_optimizer.zero_grad()
                        loss.backward()
                        right_optimizer.step()
            if id >= args.get("video_nums"):
                break

    # Save the trained models
    torch.save(left_hand_regressor.state_dict(), os.path.join("regressor", f"left_hand_global_{args.get('global')}_video_{args.get('video_nums')}_regressor.pth"))
    torch.save(right_hand_regressor.state_dict(), os.path.join("regressor", f"right_hand_global_{args.get('global')}_video_{args.get('video_nums')}_regressor.pth"))
    
    # Log the models to wandb
    wandb.save("left_hand_regressor.pth")
    wandb.save("right_hand_regressor.pth")
    
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--global', action='store', type=int, help='whether to regress the global orientaion', required=True)
    parser.add_argument('--video_nums', action='store', type=int, help='number of videos use to train the network', required=True)
    args = vars(parser.parse_args())
    main()