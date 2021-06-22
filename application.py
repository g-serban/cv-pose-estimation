import time

import cv2

import pose_module as pm


cap = cv2.VideoCapture('PoseVideos/1.mp4')
previous_time = 0
detector = pm.PoseDetector()

while True:
    success, img = cap.read()
    img = detector.find_pose(img)
    lm_list = detector.find_position(img)

    if len(lm_list) != 0:
        print(lm_list)  # print(lm_list[14]) to get info for landmark nr 14

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
