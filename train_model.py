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

from keras.callbacks import LambdaCallback
import wandb
from wandb.keras import WandbCallback
from keras.models import load_model
import argparse


#
#  This is a main code to train the Models 1 to 4.
#
#  MODEL needs to be selected.
#
#


wandb.init(project='videomelody')
config = wandb.config

MODEL = 1

from models.m1.GAN import GAN1 as GAN
# from models.m2.GAN import GAN2 as GAN
# from models.m3.GAN import GAN3 as GAN
# from models.m4.GAN import GAN4 as GAN

# Broken advance loading, does not support nested paths
# gan_versioned = __import__('models.m'+str(MODEL)+'.GAN')

# from models import *
# if args.resume:
#     model.load(wandb.checkpoint())

# --------------------------------------------------
# ---------------HYPER-PARAMETERS-------------------
# --------------------------------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy()#from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(lr = 1e-3, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lr = 1e-3, beta_1=0.5)

verbose = 0
EPOCHS = 5
BATCH_SIZE = 64
HALF_BATCH = int(BATCH_SIZE/2)

config.epochs = EPOCHS
config.batch_size = BATCH_SIZE
config.model = MODEL
# --------------------------------------------------


def log_generator(epoch, logs):
    wandb.log({'generator_loss': logs['loss'],
                     'generator_acc': logs['binary_accuracy']})

def log_discriminator(epoch, logs):
    wandb.log({
            'discriminator_loss': logs['loss'],
            'discriminator_acc': logs['binary_accuracy']})

def log_demo(epoch, vldnr):
    # gen_roll = np.zeros((val_size, pianoroll_size))
    for i in range(vldnr):
        if MODEL == 4:
            noise = np.random.normal(size=(1, random_size))
            vid_in = np.reshape(vid_valid_input[i], (1, sequence_length, video_vec_size))
            roll_in = np.reshape(easy_rol_valid[i], (1, sequence_length, pianoroll_size))
            g_roll = gan.generate([vid_in, noise, roll_in])
            g_roll = np.reshape(g_roll, (sequence_length, pianoroll_size))
        if MODEL == 1:
            noise = np.random.normal(size=(1, random_size))
            vid_in = np.reshape(vid_valid_input[i], (1, sequence_length, video_vec_size))
            g_roll = gan.generate([vid_in, noise])
            g_roll = np.reshape(g_roll, (sequence_length, pianoroll_size))
        # print(g_roll.shape)
        # print(vid_valid.shape)
        # for i in range(val_size):
        #     noise = np.random.normal(size=(val_size,random_size))
        #     vid_in = np.reshape(vid_valid_input[i], (1, sequence_length, video_vec_size))
        #     roll_in = np.reshape(gen_roll[i:i+sequence_length], (1, sequence_length, pianoroll_size))
        #     g_roll = gan.generate([vid_in, noise, roll_in])
        #     gen_roll[i+sequence_length] = g_roll
        demonizer.log_demo(str(epoch)+"_"+str(i), vid_valid[i], g_roll)

def save_models(pre):
    g, d = gan.get_models()
    g.save(os.path.join(wandb.run.dir, pre + "generator.h5"))
    d.save(os.path.join(wandb.run.dir, pre + "discriminator.h5"))
    wandb.save(pre + 'generator.h5')
    wandb.save(pre + 'discriminator.h5')
    # wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

def load_models():
    g, d = gan.get_models()
    g.save(os.path.join(wandb.run.dir, "generator.h5"))
    d.save(os.path.join(wandb.run.dir, "discriminator.h5"))
    wandb.save('generator.h5')
    wandb.save('discriminator.h5')
    wandb.save('../logs/*ckpt*')
    # wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

def train(easy_rolls, notes, img_data, roll_data, epochs, batch_size):
    bat_n = int(img_data.shape[0] / batch_size)

    wandb_logging_g = LambdaCallback(on_epoch_end=log_generator)
    wandb_logging_d = LambdaCallback(on_epoch_end=log_discriminator)


    for epoch in range(epochs):
        ep_start = time.time()
        diff_pitches_epoch = set()
        for b_nr in range(int(bat_n)):

            idx = np.random.randint(0, img_data.shape[0], BATCH_SIZE)
            img_batch = img_data[idx]
            roll_batch = roll_data[idx]
            easy_batch = easy_rolls[idx]
            note_batch = notes[idx]

            d_loss_real, d_acc_real, d_loss_fake, d_acc_fake, g_loss, g_acc, pitch_set, pitch_freq = gan.train_step(easy_batch, note_batch, img_batch, roll_batch, BATCH_SIZE, wandb_logging_g, wandb_logging_d, verbose)

            wandb.log({'D loss real': d_loss_real})
            wandb.log({'D real accuracy': d_acc_real})
            wandb.log({'D loss fake': d_loss_fake})
            wandb.log({'D fake accuracy': d_acc_fake})
            wandb.log({'G loss': g_loss})
            wandb.log({'G accuracy': g_acc})
            wandb.log({'pitch count': len(pitch_set)})
            wandb.log({'pitch average frequence': pitch_freq})
            diff_pitches_epoch = diff_pitches_epoch.union(pitch_set)

        print('Time for epoch %d is %.3f sec'% (epoch + 1, time.time()-ep_start))
        wandb.log({'pitch count EPOCH': len(diff_pitches_epoch)})
        if epoch % 5 == 0:
            log_demo(epoch,1)
            save_models(str(epoch))

    # log_demo(epoch,valid_size)
    save_models(str(""))


# LOAD INPPUT TO NETWORK --------------------------------------------------

pianoroll_size = 2502

video_vec_size = 100

sequence_length = 50

sample_size = 500
valid_size = 10
random_size = sequence_length * 128

# Create model --------------------------------------------------
gan = GAN(random_size, pianoroll_size, video_vec_size, sequence_length, generator_optimizer, discriminator_optimizer)
gan.summary()
demonizer = demonizer(wandb)

restore = False
if restore:
    if MODEL == 1:
         # dir = "run-20200505_120928-2meg365v" # MODEL 1
         dir = "run-20200517_013218-rfq7hyin"
    if MODEL == 2:
         dir = "__________" # MODEL 2
    if MODEL == 3:
         dir = "__________" # MODEL 3
    if MODEL == 4:
        # dir = "run-20200505_232951-3i0930tn" # MODEL 4.1
         dir = "run-20200506_083450-3t37r42j" # MODEL 4.2

    gen = "wandb\\" + dir + "\\generator.h5"
    dis = "wandb\\" + dir + "\\discriminator.h5"
    print("xxx")
    gan.load_weights(gen, dis)

fd = feeder(config.model, sample_size = sample_size, valid_size = valid_size)

vid_valid_input, n_valid, gen_roll, easy_rol_valid, vid_valid = fd.get_validation()

vid_input, n_input, note_input, easy_rolls = fd.get_input()

# print(vid_valid_input.shape)
# print(vid_valid.shape)
print(n_input.shape)
# print(easy_rolls.shape)
# print(vid_valid.shape)
# print(gen_roll.shape)
# print(np.max(easy_rolls))
# print(np.max(vid_input))
# print(np.max(note_input))
# print(np.max(easy_rol_valid))
# print(np.max(vid_valid_input))
# print(np.max(vid_valid))

# n_input = []
# easy_rolls = []
# n_input = np.reshape(n_input,(sample_size, roll_size))

demonizer.log_demo("true", vid_valid[0], gen_roll[0])
# print(gen_roll.shape)
# log_demo("test",1)
# log_demo("all",10)



train(easy_rolls, n_input, vid_input, note_input, EPOCHS, BATCH_SIZE)
