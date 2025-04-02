import cv2

video_cap = cv2.VideoCapture("data/_2FBDaOPYig_1-3-rgb_front.mp4")
dawe = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
dawdawee = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
for i in range(int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = video_cap.read()
    cv2.imwrite(f"render_result/frames/{i}.jpg", frame)