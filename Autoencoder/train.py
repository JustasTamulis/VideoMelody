from cae import *
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
    while (fc < frameCount-100  and ret):
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
checkpoint_dir = folder_name + "checkpoints3\\"
avi_name = 'Koyaanisqatsi.120p.avi'
image_name = 'Koyaanisqatsi.npy'
avi_path = folder_name + avi_name
image_path = folder_name + image_name

cae = ConvAutoEncoder(input_shape=(120,120,3), output_dim=100)
# cae.load_weights(prefix = "last2_")

# video = getVideo(avi_path)
video = np.load(image_path)
np.random.shuffle(video)
# plot_many(video)
FRAMES = video.shape[0]
IMG_HEIGHT = 120
IMG_WIDTH = 120
# np.save(image_path, video)
print(FRAMES)

All_train, X_validate, X_test = np.split(video, [int(0.965 * FRAMES), int(0.975 * FRAMES)])
#
video = None
cae.load_weights(prefix = "last_")


X_validate = X_validate.astype(np.float32)
X_validate = X_validate / 255


# print(X_test.shape)
# print(X_validate.shape)
# print(All_train.shape)

epochs = 1
train_size = int(len(All_train)/25)
j = 0
for i in np.arange(0, len(All_train) - train_size, train_size):
    X_train,  All_train = np.split(All_train, [train_size])
    # X_train = All_train[i:i+train_size]
    X_train = X_train.astype(np.float32)
    X_train = X_train / 255
    mse = cae.fit(X_train, X_validate, epochs=epochs)
    j = j + 1
    cae.save_weights(prefix = str(j) + "_" + str(round(mse,4)) + "_")
    print(j)
    cae.save_weights(prefix = "last_")

All_train = None
cae.save_weights(prefix = "new_")
# Test
X_test = X_test.astype(np.float32)
X_test = X_test / 255
test_codes = cae.encode(X_test)
print(np.max(test_codes))
test_reconstructed = cae.decode(test_codes)

plot_many_vs_reconstructed(X_test, test_reconstructed, n = 8)
plot_many_vs_reconstructed(X_test, test_reconstructed, n = 8)
plot_many_vs_reconstructed(X_test, test_reconstructed, n = 8)


# video = np.load(image_path)
# FRAMES = video.shape[0]
# All_train, X_validate, X_test = np.split(video, [int(0.99 * FRAMES), int(0.995 * FRAMES)])
# #
# X_test = X_test.astype(np.float32)
# X_test = X_test / 255
# test_codes = cae.encode(X_test)
# test_reconstructed = cae.decode(test_codes)
#
# plot_many_vs_reconstructed(X_test, test_reconstructed, n = 4)
