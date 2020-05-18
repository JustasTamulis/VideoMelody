from cae import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import psutil

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

def plot_many_vs_reconstructed(original, reconstructed, n=5):
    plt.figure(figsize=(12,4))
    n = [2,n]
    for i in range(n[1]):
        idx = np.random.randint(0, original.shape[0]-1)
        plt.subplot(n[0], n[1], i+1)
        plt.imshow(original[idx,:,:,:], cmap=plt.get_cmap("Greys"))
        plt.axis('off')
        plt.subplot(n[0], n[1], i+1+n[1])
        plt.imshow(reconstructed[idx,:,:,:], cmap=plt.get_cmap("Greys"))
        plt.axis('off')
    plt.show()

def plot_many(images, n=[2,6]):
    plt.figure(figsize=(12,4))
    for i in range(n[0]):
        for j in range(n[1]):
            plt.subplot(n[0], n[1], j*n[0]+i+1)
            plt.imshow(images[np.random.randint(0, images.shape[0]-1),:,:,:], cmap=plt.get_cmap("Greys"))
            plt.axis('off')
    plt.show()

folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\"
avi_name = 'video\\2.avi'
image_name = 'Koyaanisqatsi.npy'
avi_path = folder_name + avi_name
image_path = folder_name + image_name

# Create network

cae = ConvAutoEncoder(input_shape=(120,120,3), output_dim=100)
cae.load_weights(prefix = "koya_")

# Load video

video = getVideo(avi_path)
# video = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
temp = np.copy(video[:,:,:,0])
video[:,:,:,0] = video[:,:,:,2]
video[:,:,:,2] = temp
# # video = np.load(image_path)
video = video[2000:3000,:,:,:]
# # video = np.load(image_path)
# # np.random.shuffle(video)
# print(video.shape)
FRAMES = video.shape[0]
IMG_HEIGHT = 120
IMG_WIDTH = 120
print(FRAMES)
plot_many(video)
video = video.astype(np.float32)
video = video / 255
plot_many(video)

# Test

test_codes = cae.encode(video)
test_reconstructed = cae.decode(test_codes)
plot_many_vs_reconstructed(video, test_reconstructed, n = 4)
