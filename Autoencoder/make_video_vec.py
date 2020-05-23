import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import psutil
from cae import *

folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\"

for i in range(1,8):

    video_path = folder_name + "image_input\\" + str(i) + '.npy'
    vec_path = folder_name + "img_vec\\" + str(i) + '.npy'

    video = np.load(video_path)

    cae = ConvAutoEncoder(input_shape=(120,120,3), output_dim=100)
    cae.load_weights(prefix = "koya_")

    video = video.astype(np.float32)
    video = video / 255

    vec = cae.encode(video)
    print(np.max(vec))
    print(type(vec))
    print(vec.shape)
    np.save(vec_path, vec)
