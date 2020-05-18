import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import psutil
process = psutil.Process(os.getpid())
# print(process.memory_info().rss / 1000000)  # in megabytes

def getVideo(name):
    cap = cv2.VideoCapture(name)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frameCount-1  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    return buf

def plot_many(images, n=[2,6]):
    plt.figure(figsize=(12,4))
    for i in range(n[0]):
        for j in range(n[1]):
            plt.subplot(n[0], n[1], j*n[0]+i+1)
            plt.imshow(images[np.random.randint(0, images.shape[0]-1),:,:,:], cmap=plt.get_cmap("Greys"))
            plt.axis('off')
    plt.show()

folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\"

for i in range(1,8):
    avi_name = 'video\\' + str(i) + '.avi'
    image_name = 'npy\\' + str(i) + '.npy'
    avi_path = folder_name + avi_name
    image_path = folder_name + image_name
    # print(avi_path)
    video = getVideo(avi_path)
    print(video.shape)
    temp = np.copy(video[:,:,:,0])
    video[:,:,:,0] = video[:,:,:,2]
    video[:,:,:,2] = temp
    np.save(image_path, video)
