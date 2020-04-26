import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import psutil
process = psutil.Process(os.getpid())

folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\"

for i in range(1,8):

    piano_path = folder_name + "midi_npy\\" + str(i) + '.npy'
    video_path = folder_name + "image_input\\" + str(i) + '.npy'

    video = np.load(video_path)
    piano = np.load(piano_path)

    FRAMES = video.shape[0]
    roll_count = piano.shape[0]

    # Make tons of pairs: (100 img + 1 note)

    np.save(output_path, image_input)
