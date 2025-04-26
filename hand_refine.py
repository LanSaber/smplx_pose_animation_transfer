import os.path
import pickle
import json
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
import torch
import cv2
import numpy as np
import pyrender
import trimesh
import time
import gzip

class HandRender:
    def __init__(self, faces):
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)
        self.faces = faces
        self.faces_left = self.faces[:, [0, 2, 1]]
        LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
        self.mesh_base_color = LIGHT_PURPLE

    def render(self, vertices, width, height, focal_length, cam_t, is_right):
        """
        Create a pyrender scene and renderer
        """
        scene = pyrender.Scene(ambient_light=[0.6, 0.6, 0.6], bg_color=[0, 0, 0, 0])

        camera = pyrender.IntrinsicsCamera(
            fx=focal_length,
            fy=focal_length,
            cx=width / 2,
            cy=height / 2,
            zfar=1e12
        )
        camera_pose = np.eye(4)
        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
            camera_pose[:3, 3] = camera_translation
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        scene.add(light, pose=np.eye(4))

        vertex_colors = np.array([(*self.mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices.copy(), self.faces_left.copy(),
                                   vertex_colors=vertex_colors)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        mesh = pyrender.Mesh.from_trimesh(mesh)
        renderer = pyrender.OffscreenRenderer(
            viewport_width=width,
            viewport_height=height,
            point_size=1.0
        )
        scene.add(mesh, 'mesh')

        color, depth = renderer.render(scene)
        renderer.delete()
        return color, depth


    def create_hand_mesh(self, vertices, is_right=True):
        """
        Create a mesh with proper material
        """
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces if is_right else self.faces_left)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.4,
            alphaMode='BLEND',
            baseColorFactor=[0.0, 0.3, 1.0, 0.7] if not is_right else [0.0, 1.0, 0.3, 0.7]
        )

        mesh = pyrender.Mesh.from_trimesh(
            mesh,
            material=material,
            smooth=True
        )

        return mesh


if __name__ == '__main__':

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)

    # Create output directories
    os.makedirs("visualization_mesh", exist_ok=True)
    os.makedirs("visualization_videos", exist_ok=True)

    # Load MANO faces
    mano_faces = pipe.wilor_model.mano.faces
    hand_render = HandRender(mano_faces)

    # Open input video
    # input_video_path = "data/_2FBDaOPYig_1-3-rgb_front.mp4"
    input_video_path = "render_result/vid_ego/output_video.mp4"
    vid = cv2.VideoCapture(input_video_path)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    hand_data_list = []

    # Get video properties
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # Create video writers
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_name = os.path.basename(input_video_path).split('.')[0]

    # Video writer for visualization
    out_vis = cv2.VideoWriter(
        f"visualization_videos/{video_name}_hand_mesh_{timestamp}.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    print(f"Processing {total_frames} frames...")
    start_time = time.time()

    for i in range(total_frames):
        vid.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, img_per_frame = vid.read()
        if not ret:
            continue

        image = cv2.cvtColor(img_per_frame, cv2.COLOR_BGR2RGB)
        outputs = pipe.predict(image)
        hand_data = {}

        colors = np.zeros((height, width, 3), dtype=np.uint8)
        masks = np.zeros((height, width), dtype=np.bool_)
        # Create renderer with current frame's focal length

        for output in outputs:
            vertices = output["wilor_preds"]["pred_vertices"][0]
            pred_cam = output["wilor_preds"]["pred_cam"][0]
            cam_t = output["wilor_preds"]['pred_cam_t_full'][0]
            focal_length = outputs[0]["wilor_preds"]["scaled_focal_length"]
            color, depth = hand_render.render(width=width, height=height, focal_length=focal_length, cam_t=cam_t, vertices=vertices, is_right=(output["is_right"] == 1.0))


            if output["is_right"] == 0.0:
                hand_data["smplx_lhand_pose"] = output["wilor_preds"]["hand_pose"].reshape((-1)).tolist()
                hand_data["lhand_pred_vertices"] = output["wilor_preds"]["pred_vertices"].reshape((-1)).tolist()
                hand_data["lwrist_global_orient"] = output["wilor_preds"]["global_orient"].reshape((-1)).tolist()
            else:
                hand_data["smplx_rhand_pose"] = output["wilor_preds"]["hand_pose"].reshape((-1)).tolist()
                hand_data["rhand_pred_vertices"] = output["wilor_preds"]["pred_vertices"].reshape((-1)).tolist()
                hand_data["rwrist_global_orient"] = output["wilor_preds"]["global_orient"].reshape((-1)).tolist()
            # Render
            mask = depth > 0
            colors[mask] = color[mask]
            masks = masks | mask

        # Create visualization frame
        vis_frame = image.copy()
        vis_frame[masks] = colors[masks].astype(np.uint8)

        # Convert to BGR for saving
        vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

        # Add frame counter
        cv2.putText(vis_frame_bgr, f"Frame: {i}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Write frames to videos
        out_vis.write(vis_frame_bgr)

        # Save individual frames (optional)
        if i % 30 == 0:
            cv2.imwrite(f"visualization_mesh/frame_{i:04d}.jpg", vis_frame_bgr)

        # Print progress
        if i % 100 == 0:
            elapsed_time = time.time() - start_time
            frames_per_second = (i + 1) / elapsed_time
            estimated_time_left = (total_frames - i - 1) / frames_per_second
            print(f"Processed {i}/{total_frames} frames "
                  f"({frames_per_second:.2f} fps, "
                  f"ETA: {estimated_time_left / 60:.1f} minutes)")

        hand_data_list.append(hand_data)

    # Release all resources
    vid.release()
    out_vis.release()

    total_time = time.time() - start_time
    print(f"\nProcessing completed in {total_time / 60:.1f} minutes")
    print(f"Output videos saved in visualization_videos/")

    # Save hand pose data
    file_name = os.path.join("pose_data", f"{video_name}_hand_refine.pkl")
    with open(file_name, "wb") as f:
        pickle.dump(hand_data_list, f)
