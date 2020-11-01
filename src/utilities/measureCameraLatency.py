"""
Run this script then
point the camera to look at the window,
watch the color flips between black and white.
Slightly increase "THRESHOLD" value if it doesn't flip.
"""

import cv2.cv2 as cv2
import numpy as np

# Initialize USB webcam feed
CAM_INDEX = 1
# Adjust this value if it doesn't flip. 0~255
THRESHOLD = 100
# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720

### USB webcam ###
camera = cv2.VideoCapture(CAM_INDEX)
if ((camera == None) or (not camera.isOpened())):
    print('\n\n')
    print('Error - could not open video device.')
    print('\n\n')
    exit(0)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, IM_WIDTH)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_HEIGHT)
# # save the actual dimensions
# actual_video_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
# actual_video_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print('actual video resolution:{:.0f}x{:.0f}'.format(actual_video_width, actual_video_height))

prev_tick = cv2.getTickCount()
frame_number, prev_change_frame = 0, 0
is_dark = False

time_intervals = []

while True:
    frame_number += 1

    _, frame = camera.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    is_now_dark = np.average(img) < THRESHOLD

    if is_dark != is_now_dark:
        is_dark = is_now_dark
        new = cv2.getTickCount()

        print("{:.3f} seconds ({:.0f} frames)".format(
            (new - prev_tick) / cv2.getTickFrequency(),
            frame_number - prev_change_frame,
        ))

        time_intervals.insert(0, (new - prev_tick)/cv2.getTickFrequency())
        
        prev_tick = new

        prev_change_frame = frame_number

        fill_color = 255 if is_dark else 0
        show = np.full(img.shape, fill_color, dtype=img.dtype)

        cv2.imshow('frame', show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print('\nAverage latency: {:.5f}'.format(sum(time_intervals)/len(time_intervals)))