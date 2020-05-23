# Simple convolutional model.
# Input - image from a clip.
# Output - 0->1, probabily that given image is from clip, rather that random data.
#
#
#



# from __future__ import absolute_import, divivideo_lenon, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# from os import startfile
# from moviepy.editor import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


def getVideo(name):
    cap = cv2.VideoCapture(name)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    # cap.release()
    # cv2.namedWindow('frame 10')
    # cv2.imshow('frame 10', buf[10])
    # cv2.show()
    # cv2.waitKey(0)

    return buf

def showImage(image):
    plt.imshow(image, cmap='gray')
    plt.show()

# tf.enable_eager_execution()
video_name = 'HarderBetterFasterStronger.mp4'
video = getVideo(video_name)

FRAMES = video.shape[0]
IMG_HEIGHT = 360
IMG_WIDTH = 480

random_examples = 1000
noise = np.random.uniform(0,1,(random_examples,IMG_HEIGHT,IMG_WIDTH,3))
noise = noise.reshape(noise.shape[0], IMG_HEIGHT, IMG_WIDTH, 3).astype('float32')
video = video.reshape(FRAMES, IMG_HEIGHT, IMG_WIDTH, 3).astype('float32')
video = video / 255.0
[video_train, video_test] = np.split(video, [int(FRAMES * 0.8)])

video_len = video.shape[0]
x_train = np.concatenate((video_train,noise), axis=0)
y_train = np.concatenate((np.ones(video_train.shape[0]),np.zeros(noise.shape[0])))

random_examples = 100
noise = np.random.uniform(0,1,(random_examples,IMG_HEIGHT,IMG_WIDTH,3))
noise = noise.reshape(noise.shape[0], IMG_HEIGHT, IMG_WIDTH, 3).astype('float32')

x_test = np.concatenate((video_test,noise), axis=0)
y_test = np.concatenate((np.ones(video_test.shape[0]),np.zeros(noise.shape[0])))

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH,3)),
#   tf.keras.layers.Dense(IMG_HEIGHT, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(1, activation='softmax')
# ])

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# predictions = model(video[:1]).numpy()
# print(predictions)


model.fit(x_train, y_train, epochs=1)

score = model.evaluate(x_test,  y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# INFORMATION print

# print(noise.shape)
# print(video.shape)
# print(np.amin(video))
# print(np.amax(video))
# print(noise.dtype)
# print(video.dtype)
