import cv2
import numpy as np
import json
import glob
import os


def draw_pose_from_json(json_file, image_size=(640, 480), conf_threshold=0.1):
    """
    Loads an OpenPose JSON file, scales the keypoints to fit a canvas of image_size,
    draws the body and hand poses on the canvas, and returns the image.
    """
    # Create a blank canvas with the desired image size (width, height)
    canvas = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # If no person is detected, return the blank canvas.
    if not data.get("people"):
        return canvas

    # Use only the first person detected.
    person = data["people"][0]

    # --------------------------
    # Process Body Keypoints
    # --------------------------
    body_keypoints = person.get("pose_keypoints_2d", [])
    if not body_keypoints:
        return canvas

    # Reshape to (num_keypoints, 3) for x, y, confidence.
    body_keypoints = np.array(body_keypoints).reshape(-1, 3)

    # Determine valid keypoints (with confidence above threshold).
    valid_body = body_keypoints[:, 2] > conf_threshold
    if not np.any(valid_body):
        return canvas

    valid_x = body_keypoints[valid_body, 0]
    valid_y = body_keypoints[valid_body, 1]
    min_x, max_x = valid_x.min(), valid_x.max()
    min_y, max_y = valid_y.min(), valid_y.max()

    # Compute a uniform scale factor to fit the skeleton in the canvas
    if max_x - min_x < 1 or max_y - min_y < 1:
        scale = 1.0
    else:
        scale_x = image_size[0] / (max_x - min_x)
        scale_y = image_size[1] / (max_y - min_y)
        scale = min(scale_x, scale_y) * 0.9  # use 90% to add a margin

    # Compute offsets to center the skeleton.
    skeleton_width = (max_x - min_x) * scale
    skeleton_height = (max_y - min_y) * scale
    offset_x = (image_size[0] - skeleton_width) / 2 - min_x * scale
    offset_y = (image_size[1] - skeleton_height) / 2 - min_y * scale

    # Scale the body keypoints.
    scaled_body = []
    for (x, y, conf) in body_keypoints:
        new_x = int(x * scale + offset_x)
        new_y = int(y * scale + offset_y)
        scaled_body.append((new_x, new_y, conf))

    # Draw body keypoints.
    for (x, y, conf) in scaled_body:
        if conf > conf_threshold:
            cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)

    # Define body skeleton connections (example for BODY_25 model).
    body_skeleton = [
        (1, 2), (1, 5), (2, 3), (3, 4),
        (5, 6), (6, 7), (1, 8), (8, 9),
        (9, 10), (10, 11), (8, 12), (12, 13),
        (13, 14), (1, 0), (0, 15), (15, 17),
        (0, 16), (16, 18), (14, 19), (19, 20),
        (14, 21), (11, 22), (22, 23), (11, 24)
    ]

    # Draw body skeleton.
    for i, j in body_skeleton:
        if i < len(scaled_body) and j < len(scaled_body):
            x1, y1, conf1 = scaled_body[i]
            x2, y2, conf2 = scaled_body[j]
            if conf1 > conf_threshold and conf2 > conf_threshold:
                cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # --------------------------
    # Process Hand Keypoints
    # --------------------------
    # Define hand skeleton connections for a 21-keypoint hand.
    hand_skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
    ]

    def draw_hand(key, color_circle, color_line):
        """Helper function to process and draw a hand pose."""
        hand_data = person.get(key, [])
        if not hand_data:
            return

        # Reshape the flat list into (21, 3).
        hand_keypoints = np.array(hand_data).reshape(-1, 3)
        # Scale and offset hand keypoints using the same transformation as the body.
        scaled_hand = []
        for (x, y, conf) in hand_keypoints:
            new_x = int(x * scale + offset_x)
            new_y = int(y * scale + offset_y)
            scaled_hand.append((new_x, new_y, conf))

        # Draw hand keypoints.
        for (x, y, conf) in scaled_hand:
            if conf > conf_threshold:
                cv2.circle(canvas, (x, y), 3, color_circle, -1)

        # Draw hand skeleton.
        for i, j in hand_skeleton:
            if i < len(scaled_hand) and j < len(scaled_hand):
                x1, y1, conf1 = scaled_hand[i]
                x2, y2, conf2 = scaled_hand[j]
                if conf1 > conf_threshold and conf2 > conf_threshold:
                    cv2.line(canvas, (x1, y1), (x2, y2), color_line, 2)

    # Draw left hand (if available) in a distinct color (e.g., yellow).
    draw_hand("hand_left_keypoints_2d", color_circle=(0, 255, 255), color_line=(0, 200, 200))
    # Draw right hand (if available) in another color (e.g., magenta).
    draw_hand("hand_right_keypoints_2d", color_circle=(255, 0, 255), color_line=(200, 0, 200))

    return canvas


def create_animation_from_json(json_folder, output_video, fps=12, image_size=(640, 480)):
    """
    Reads all JSON files in the specified folder, draws the body and hand pose for each frame,
    and writes the frames into an MP4 video.
    """
    # Get a sorted list of JSON files.
    json_files = sorted(glob.glob(os.path.join(json_folder, "*.json")))

    if not json_files:
        print("No JSON files found in the specified folder.")
        return

    # Initialize the video writer with the specified codec, frame rate, and size.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, image_size)

    print(f"Processing {len(json_files)} JSON files...")
    for json_file in json_files:
        frame = draw_pose_from_json(json_file, image_size=image_size)
        video_writer.write(frame)

    video_writer.release()
    print(f"Animation video saved to: {output_video}")

if __name__ == '__main__':
    # Update these paths as needed.
    name='_2FBDaOPYig_1-3-rgb_front'
    json_folder = f"D:\how2sign\openpose_output\json\{name}"  # Folder containing your OpenPose JSON files.
    output_video = f"{name}_openpose_animation_secondpart.mp4"  # Output video file.
    if name.endswith(".pkl"):
        with open(json_folder, "rb") as f:
            import pickle
            file = pickle.load(f)
            e = 0

    # Create the animation video.
    create_animation_from_json(json_folder, output_video, fps=6, image_size=(640, 480))