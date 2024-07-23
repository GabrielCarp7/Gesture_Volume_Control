import cv2
import time
import numpy as np
import hand_traking_module
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

camera_width, camera_height = 640, 480

cap = cv2.VideoCapture(1)
cap.set(3, camera_width)
cap.set(4, camera_height)

previous_time = 0

detector = hand_traking_module.HandDetector(detection_conf=0.7)

# Initialization
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Volume controls
# volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()

min_volume = volume_range[0]
max_volume = volume_range[1]

volume = 0
volume_bar = 400
volume_percentage = 0


while True:
    success, img = cap.read()
    img = detector.find_hands(img)

    landmark_list = detector.find_pos(img, draw=False)

    if landmark_list != 0:
        # Get landmark positions for thumb tip and index fingertip
        x1, y1 = landmark_list[4][1], landmark_list[4][2]
        x2, y2 = landmark_list[8][1], landmark_list[8][2]
        # Find the center between the two landmarks
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        # Draw a circle over the two landmarks
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

        # Connect the two landmarks with a line
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Draw a circle in the middle of the line
        cv2.circle(img, (center_x, center_y), 15, (255, 0, 255), cv2.FILLED)

        # Get the lengths between the two fingers
        length = math.hypot(x2 - x1, y2 - y1)

        # Convert the hand range into volume range
        volume = np.interp(length, [50, 300], [min_volume, max_volume])
        volume_bar = np.interp(length, [50, 300], [400, 150])
        volume_percentage = np.interp(length, [50, 300], [0, 100])

        volume.SetMasterVolumeLevel(volume, None)

        if length < 50 or length > 299:
            cv2.circle(img, (center_x, center_y), 15, (0, 255, 0), cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, (int(volume_bar))), (85, 400), (0, 255, 0), cv2.FILLED)

        cv2.putText(img, f"{int(volume_percentage)}%", (40, 450), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
