import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import psutil
process = psutil.Process(os.getpid())

folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\"

for i in range(1,8):

    npy_path = folder_name + 'video_npy\\' + str(i) + '.npy'
    piano_path = folder_name + "midi_npy\\" + str(i) + '.npy'

    output_path = folder_name + "image_input\\" + str(i) + '.npy'

    video = np.load(npy_path)
    piano = np.load(piano_path)
    print(i)
    print(video.shape)
    print(piano.shape)

    FRAMES = video.shape[0]
    roll_count = piano.shape[0]

    image_input = []
    fps_diff = FRAMES / roll_count
    for i in range(roll_count):
        idx = min(int((i+1) * fps_diff), FRAMES-1)
        image_in = video[idx ,:,:,:]
        image_input.append(image_in)
    image_input = np.reshape(image_input, (roll_count, 120, 120, 3))
    print(image_input.shape)

    np.save(output_path, image_input)
