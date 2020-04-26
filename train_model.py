import os
import random
import time
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pygame
import pretty_midi
from moviepy.editor import *
from moviepy.audio.AudioClip import AudioArrayClip
from pypianoroll import Multitrack, Track
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.utils import plot_model
import tensorflow as tf

from models import *
from log import demonizer
from Data import *

from keras.callbacks import LambdaCallback
import wandb
from wandb.keras import WandbCallback
from keras.models import load_model
import argparse
#
wandb.init(project='videomelody')
config = wandb.config

# --------------------------------------------------
# ---------------HYPER-PARAMETERS-------------------
# --------------------------------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy()#from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(lr = 1e-2)#, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lr = 1e-2)#, beta_1=0.5)

EPOCHS = 3
BATCH_SIZE = 128
HALF_BATCH = int(BATCH_SIZE/2)

# config.epochs = 3
# config.discriminator_epochs = 1
# config.discriminator_examples = 10000
# config.generator_epochs = 12
# config.generator_examples = 10000
# config.batch_size = 256
# config.half_batch_size = 128
# --------------------------------------------------


def log_generator(epoch, logs):
    wandb.log({'generator_loss': logs['loss'],
                     'generator_acc': logs['acc']})

def log_discriminator(epoch, logs):
    wandb.log({
            'discriminator_loss': logs['loss'],
            'discriminator_acc': logs['acc']})

def log_demo(epoch):
    noise = np.random.normal(size=(len(vid_valid_input),100))
    gen_roll = gan.generate([vid_valid_input, noise])
    demonizer.log_demo(epoch, vid_valid, gen_roll)

def train(img_data, roll_data, epochs, batch_size):
    bat_n = int(img_data.shape[0] / batch_size)

    wandb_logging_g = LambdaCallback(on_epoch_end=log_generator)
    wandb_logging_d = LambdaCallback(on_epoch_end=log_discriminator)

    for epoch in range(epochs):
        ep_start = time.time()
        for b_nr in range(int(bat_n)):

            idx = np.random.randint(0, img_data.shape[0], BATCH_SIZE)
            img_batch = img_data[idx]
            roll_batch = roll_data[idx]

            gan.train_step(img_batch, roll_batch, BATCH_SIZE, wandb_logging_g, wandb_logging_d, verbose)

        print('Time for epoch %d is %.3f sec'% (epoch + 1, time.time()-ep_start))
        log_demo(epoch)


# LOAD INPPUT TO NETWORK --------------------------------------------------

pianoroll_size = 2502
random_size=100
video_vec_size = 100
num_video_vec = 50
sample_size = 3500
fd = feeder(sample_size = sample_size, num_video_vec = num_video_vec, valid_size = 200)
demonizer = demonizer(wandb)

vid_input, note_input, vid_valid_input, vid_valid = fd.gen_input()

# print(len(vid_input))
# print(vid_input.shape)
# print(len(note_input))
# print(note_input.shape)

# Create model --------------------------------------------------
print("CREATING MODEL")
gan = GAN(random_size, pianoroll_size, (num_video_vec, video_vec_size), generator_optimizer, discriminator_optimizer)
verbose = 0
print("TRAINING MODEL")
train(vid_input, note_input, EPOCHS, BATCH_SIZE)
