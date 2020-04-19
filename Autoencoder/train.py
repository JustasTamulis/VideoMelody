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

folder_name = "F:\\CompSci\\project\\Data\\Koyatsiqatsi\\"
checkpoint_dir = folder_name + "checkpoints3\\"
avi_name = 'Koyaanisqatsi.120p.avi'
image_name = 'Koyaanisqatsi.npy'
avi_path = folder_name + avi_name
image_path = folder_name + image_name


folder_name2 = "F:\\CompSci\\project\\Data\\Piano midi\\"
image_name2 = 'image_ndarray.npy'
image_path2 = folder_name2 + image_name2

cae = ConvAutoEncoder(input_shape=(120,120,3), output_dim=100)
# cae.load_weights(prefix = "last2_")

# video = getVideo(avi_path)
video = np.load(image_path2)
np.random.shuffle(video)
# print(process.memory_info().rss / 1000000)  # in megabytes
# plot_many(video)
FRAMES = video.shape[0]
IMG_HEIGHT = 120
IMG_WIDTH = 120
# np.save(image_path, video)
print(FRAMES)
epochs = 15
All_train, X_validate, X_test = np.split(video, [int(0.9 * FRAMES), int(0.95 * FRAMES)])
#
video = None
mse = cae.fit(All_train, X_validate, epochs=epochs)
cae.save_weights(prefix = "viz_")
# X_test = X_test.astype(np.float32)
# X_test = X_test / 255
# X_validate = X_validate.astype(np.float32)
# X_validate = X_validate / 255
# # print(process.memory_info().rss / 1000000)  # in megabytes
#
# train_size = 0.025
# j = 0
# for i in np.arange(train_size, 0.94, train_size):
#     X_train,  All_train = np.split(All_train, [int(train_size * FRAMES)])
#     X_train = X_train.astype(np.float32)
#     X_train = X_train / 255
#     mse = cae.fit(X_train, X_validate, epochs=epochs)
#     j = j + 1
#     # cae.save_weights(prefix = str(j) + "_" + str(round(mse,4)) + "_")
#     print(j)
#     cae.save_weights(prefix = "last3_")

# Test
test_codes = cae.encode(X_test)
test_reconstructed = cae.decode(test_codes)

plot_many_vs_reconstructed(X_test, test_reconstructed, n = 4)



video = np.load(image_path)
FRAMES = video.shape[0]
All_train, X_validate, X_test = np.split(video, [int(0.99 * FRAMES), int(0.995 * FRAMES)])
#
X_test = X_test.astype(np.float32)
X_test = X_test / 255
test_codes = cae.encode(X_test)
test_reconstructed = cae.decode(test_codes)

plot_many_vs_reconstructed(X_test, test_reconstructed, n = 4)
