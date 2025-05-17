import sys
import cv2
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import os
from animation import *
import pickle
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Data Visualization")
        self.setGeometry(100, 100, 1200, 600)  # Increased window width

        # Initialize video captures
        self.cap1 = None
        self.cap2 = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)

        self.pose_offset = 0

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create horizontal layout for image labels
        image_layout = QHBoxLayout()
        main_layout.addLayout(image_layout)

        # Create first image display label
        self.image_label1 = QLabel()
        self.image_label1.setAlignment(Qt.AlignCenter)
        self.image_label1.setMinimumSize(480, 360)
        self.image_label1.setStyleSheet("border: 1px solid black;")
        image_layout.addWidget(self.image_label1)

        # Create second image display label
        self.image_label2 = QLabel()
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label2.setMinimumSize(480, 360)
        self.image_label2.setStyleSheet("border: 1px solid black;")
        image_layout.addWidget(self.image_label2)

        # Create button layout
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        # Create video selection buttons
        self.select_button1 = QPushButton("Select Video")
        self.select_button1.clicked.connect(lambda: self.select_video(1))
        button_layout.addWidget(self.select_button1)

        # self.select_button2 = QPushButton("Select Video 2")
        # self.select_button2.clicked.connect(lambda: self.select_video(2))
        # button_layout.addWidget(self.select_button2)

    def select_video(self, label_num):
        file_directory, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Video File {label_num}",
            "",
            "Video Files (*.mp4 *.avi *.mkv);;All Files (*.*)"
        )
        if file_directory:
            file_name = os.path.basename(file_directory)
            pure_file_name = os.path.splitext(file_name)[0]
            pose_data_file = os.path.join("pose_data", pure_file_name+".pkl")
            if self.cap1 is not None:
                self.cap1.release()
            self.cap1 = cv2.VideoCapture(file_directory)
            if not self.cap1.isOpened():
                print(f"Error: Could not open video file {file_directory}")
                return
            if not os.path.exists(pose_data_file):
                print(f"Error: Could not find pose data file {pose_data_file}")
            total_frames = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Total number of frames in the video: {total_frames}")

            self.pose_offset = 0

            with open(pose_data_file, "rb") as f:
                self.pose_dicts = pickle.load(f)
                frame_num = self.pose_dicts["smplx_lhand_pose"].shape[0]
                rend_data = []
                for i in range(frame_num):
                    pose_dict = {}
                    pose_dict["smplx_lhand_pose"] = self.pose_dicts["smplx_lhand_pose"][i]
                    pose_dict["smplx_rhand_pose"] = self.pose_dicts["smplx_rhand_pose"][i]
                    pose_dict["smplx_root_pose"] = self.pose_dicts["smplx_root_pose"][i]
                    pose_dict["smplx_body_pose"] = self.pose_dicts["smplx_body_pose"][i]
                    pose_dict["smplx_jaw_pose"] = self.pose_dicts["smplx_jaw_pose"][i]
                    rend_data.append(pose_dict)
            set_pose_data(rend_data)

            # Start timer if it's not already running
            if not self.timer.isActive():
                # Get FPS from first loaded video
                cap = self.cap1 if self.cap1 is not None else self.cap2
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_delay = int(1000 / fps)
                self.timer.start(frame_delay)

    def update_frames(self):
        # Update first video frame
        if self.cap1 is not None and self.cap1.isOpened():
            ret, frame = self.cap1.read()
            if self.cap1 is None:
                self.pose_offset = 0
                self.timer.stop()
                return
            current_frame_number = int(self.cap1.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if current_frame_number in self.pose_dicts["body_bad_frames"] or current_frame_number in self.pose_dicts["hand_bad_frames"]:
                return
            if ret:
                self.display_frame(frame, self.image_label1)
                frame2 = get_image(self.pose_offset)
                self.pose_offset += 1
                self.display_frame(frame2, self.image_label2)
            else:
                self.cap1.release()
                self.cap1 = None
                self.image_label1.clear()


        # Stop timer if both videos are finished
        if self.cap1 is None and self.cap2 is None:
            self.pose_offset = 0
            self.timer.stop()

    def display_frame(self, frame, label):
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale image to fit label while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Display the frame
        label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        # Clean up resources when closing the window
        if self.cap1 is not None:
            self.cap1.release()
        if self.cap2 is not None:
            self.cap2.release()
        self.timer.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
