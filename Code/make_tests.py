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

from log import demonizer
from Data import *
from keras.models import load_model
import argparse

from models.m1.GAN import GAN1
from models.m4.GAN import GAN


# Function to convert piano roll to MIDI
def roll2midi(roll):
    generated_roll = np.zeros((len(roll),128))
    for i in range(len(roll)):
        nr = np.argmax(roll[i])
        gen_roll = hot2piano[nr]
        generated_roll[i] = gen_roll
    track = Track(pianoroll=generated_roll, program=0, is_drum=False)
    multitrack = Multitrack(tracks=[track], beat_resolution=25, tempo=60)
    pm = multitrack.to_pretty_midi()
    for notes in [instrument.notes for instrument in pm.instruments]:
        for note in notes:
            note.velocity = 100
    return pm

MODEL = 1



folder_results = "C:\\VideoMelody\\results\\" + str(MODEL) +"\\"

hot2piano_path = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\OneHot\\hot2piano.npy"
hot2piano = np.load(hot2piano_path)

pianoroll_size = 2502
video_vec_size = 100
sequence_length = 50
random_size = sequence_length * 128

valid_size = 140

# Create model --------------------------------------------------
# if MODEL != 0:
#     puzzle = __import__('models.m'+str(MODEL)+'.GAN')
if MODEL ==1:
    gan = GAN1(random_size, pianoroll_size, video_vec_size, sequence_length)
if MODEL ==2:
    gan = GAN4(random_size, pianoroll_size, video_vec_size, sequence_length)
if MODEL ==3:
    gan = GAN4(random_size, pianoroll_size, video_vec_size, sequence_length)
if MODEL ==4:
    gan = GAN4(random_size, pianoroll_size, video_vec_size, sequence_length)

# restore the best model
if MODEL == 1:
     # dir = "run-20200505_120928-2meg365v" # MODEL 1
     # dir = "run-20200514_115218-jfv2kp2r"
     dir = "run-20200517_013218-rfq7hyin"
if MODEL == 2:
     dir = "run-20200514_115218-jfv2kp2r" # MODEL 2
if MODEL == 3:
     dir = "run-20200511_113618-jfv2ko2y" # MODEL 3
if MODEL == 4:
    # dir = "run-20200505_232951-3i0930tn" # MODEL 4.1
     dir = "run-20200506_083450-3t37r42j" # MODEL 4.2
if MODEL != 0:
    gen = "wandb\\" + dir + "\\generator.h5"
    dis = "wandb\\" + dir + "\\discriminator.h5"
    gan.load_weights(gen, dis)


fd = feeder(MODEL, sample_size = 0, valid_size = valid_size)
vid_valid_input, n_valid, gen_roll, easy_rol_valid, vid_valid = fd.get_validation()

if MODEL == 0:
    for i in range(valid_size):
        # print(gen_roll[i].shape)
        midi = roll2midi(gen_roll[i])
        # print(midi.get_end_time())
        midi.write(folder_results + str(i) + ".mid")
if MODEL == 1:
    for i in range(valid_size):
        noise = np.random.normal(size=(1, random_size))
        vid_in = np.reshape(vid_valid_input[i], (1, sequence_length, video_vec_size))
        g_roll = gan.generate([vid_in, noise])
        g_roll = np.reshape(g_roll, (sequence_length, pianoroll_size))
        midi = roll2midi(g_roll)
        midi.write(folder_results + str(i) + ".mid")
if MODEL == 2:
    for i in range(valid_size):
        noise = np.random.normal(size=(1, random_size))
        vid_in = np.reshape(vid_valid_input[i], (1, sequence_length, video_vec_size))
        g_roll = gan.generate([vid_in, noise, n_valid])
        g_roll = np.reshape(g_roll, (sequence_length, pianoroll_size))
        midi = roll2midi(g_roll)
        midi.write(folder_results + str(i) + ".mid")
if MODEL == 3:
    for i in range(valid_size):
        noise = np.random.normal(size=(1, random_size))
        vid_in = np.reshape(vid_valid_input[i], (1, sequence_length, video_vec_size))
        roll_in = np.reshape(easy_rol_valid[i], (1, sequence_length, pianoroll_size))
        g_roll = gan.generate([vid_in, noise, roll_in])
        g_roll = np.reshape(g_roll, (sequence_length, pianoroll_size))
        midi = roll2midi(g_roll)
        midi.write(folder_results + str(i) + ".mid")
if MODEL == 4:
    for i in range(valid_size):
        noise = np.random.normal(size=(1, random_size))
        vid_in = np.reshape(vid_valid_input[i], (1, sequence_length, video_vec_size))
        roll_in = np.reshape(easy_rol_valid[i], (1, sequence_length, pianoroll_size))
        g_roll = gan.generate([vid_in, noise, roll_in])
        g_roll = np.reshape(g_roll, (sequence_length, pianoroll_size))
        midi = roll2midi(g_roll)
        midi.write(folder_results + str(i) + ".mid")
        # print(midi.get_end_time())



# else:
#     print("whop")
    # print(g_roll.shape)
    # print(vid_valid.shape)
    # for i in range(val_size):
    #     noise = np.random.normal(size=(val_size,random_size))
    #     vid_in = np.reshape(vid_valid_input[i], (1, sequence_length, video_vec_size))
    #     roll_in = np.reshape(gen_roll[i:i+sequence_length], (1, sequence_length, pianoroll_size))
    #     g_roll = gan.generate([vid_in, noise, roll_in])
    #     gen_roll[i+sequence_length] = g_roll
