import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('F:\\CompSci\\project\\MIDI\\DaftPunk\\OneMoreTime\\OneMoreTime.mp4')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

fc = 0
ret = True

while (fc < frameCount  and ret):
    ret, buf[fc] = cap.read()
    fc += 1

cap.release()

# cv2.namedWindow('frame 10')
# cv2.imshow('frame 10', buf[150])
# # cv2.show()
#
# cv2.waitKey(0)


#
# rootdir = 'F:\\CompSci\\project\\MIDI\\DaftPunk\\'
#
# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         print(os.path.join(subdir, file))
