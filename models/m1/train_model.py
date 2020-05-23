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
# wandb.init(project='videomelody', ckpt_config={"glob": "*ckpt*", "last": 3, "best": "*best*"})
wandb.init(project='videomelody')
config = wandb.config

# if args.resume:
#     model.load(wandb.checkpoint())

# --------------------------------------------------
# ---------------HYPER-PARAMETERS-------------------
# --------------------------------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy()#from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(lr = 1e-5)#, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lr = 1e-5)#, beta_1=0.5)

verbose = 0
EPOCHS = 20
BATCH_SIZE = 256
HALF_BATCH = int(BATCH_SIZE/2)

config.epochs = EPOCHS
config.batch_size = BATCH_SIZE
config.model = 1
# --------------------------------------------------


def log_generator(epoch, logs):
    wandb.log({'generator_loss': logs['loss'],
                     'generator_acc': logs['acc']})

def log_discriminator(epoch, logs):
    wandb.log({
            'discriminator_loss': logs['loss'],
            'discriminator_acc': logs['acc']})

def log_demo(epoch):
    # gen_roll = np.zeros((val_size, pianoroll_size))
    noise = np.random.normal(size=(1, random_size))
    vid_in = np.reshape(vid_valid_input[0], (1, sequence_length, video_vec_size))
    g_roll = gan.generate([vid_in, noise])
    g_roll = np.reshape(g_roll, (sequence_length, pianoroll_size))
    print(g_roll.shape)
    print(vid_valid.shape)
    # for i in range(val_size):
    #     noise = np.random.normal(size=(val_size,random_size))
    #     vid_in = np.reshape(vid_valid_input[i], (1, sequence_length, video_vec_size))
    #     roll_in = np.reshape(gen_roll[i:i+sequence_length], (1, sequence_length, pianoroll_size))
    #     g_roll = gan.generate([vid_in, noise, roll_in])
    #     gen_roll[i+sequence_length] = g_roll
    demonizer.log_demo(epoch, vid_valid, g_roll)

def save_models():
    g, d = gan.get_models()
    g.save(os.path.join(wandb.run.dir, "generator.h5"))
    d.save(os.path.join(wandb.run.dir, "discriminator.h5"))
    wandb.save('generator.h5')
    wandb.save('discriminator.h5')
    # wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

def load_models():
    g, d = gan.get_models()
    g.save(os.path.join(wandb.run.dir, "generator.h5"))
    d.save(os.path.join(wandb.run.dir, "discriminator.h5"))
    wandb.save('generator.h5')
    wandb.save('discriminator.h5')
    wandb.save('../logs/*ckpt*')
    # wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

def train(easy_rolls, img_data, roll_data, epochs, batch_size):
    bat_n = int(img_data.shape[0] / batch_size)

    wandb_logging_g = LambdaCallback(on_epoch_end=log_generator)
    wandb_logging_d = LambdaCallback(on_epoch_end=log_discriminator)

    for epoch in range(epochs):
        ep_start = time.time()
        for b_nr in range(int(bat_n)):

            idx = np.random.randint(0, img_data.shape[0], BATCH_SIZE)
            img_batch = img_data[idx]
            roll_batch = roll_data[idx]
            # roll_batch = []
            # easy_batch = easy_rolls[idx]
            easy_batch = []
            d_loss_real, d_loss_fake, g_loss, pitch_count = gan.train_step(easy_batch, img_batch, roll_batch, BATCH_SIZE, wandb_logging_g, wandb_logging_d, verbose)


            wandb.log({'D loss real': d_loss_real})
            wandb.log({'D loss fake': d_loss_fake})
            wandb.log({'G loss': g_loss})
            # wandb.log({'D accuracy real': d_acc_real})
            # wandb.log({'D accuracy fake': d_acc_fake})
            # wandb.log({'G accuracy': g_acc})
            wandb.log({'pitch count': pitch_count})

        print('Time for epoch %d is %.3f sec'% (epoch + 1, time.time()-ep_start))
        if epoch % 5 == 0 or epoch == EPOCHS-1:
            log_demo(epoch)
            save_models()


# LOAD INPPUT TO NETWORK --------------------------------------------------

pianoroll_size = 2502

video_vec_size = 100
sequence_length = 50
sample_size = 14000
valid_size = 1
random_size = sequence_length * 64

# Create model --------------------------------------------------
gan = GAN(random_size, pianoroll_size, video_vec_size, sequence_length, generator_optimizer, discriminator_optimizer)
# gan.summary()

restore = True
if restore:
    # restore the best model
    dir = "run-20200505_120928-2meg365v"
    gen = "wandb\\" + dir + "\\generator.h5"
    dis = "wandb\\" + dir + "\\discriminator.h5"
    print("xxx")
    gan.load_weights(gen, dis)

fd = feeder(config.model, sample_size = sample_size, valid_size = valid_size)

vid_valid_input, n_valid, gen_roll, easy_rol_valid, vid_valid = fd.get_validation()
vid_input, n_input, note_input, easy_rolls = fd.get_input()

n_input = []
easy_rolls = []

demonizer = demonizer(wandb)
# demonizer.log_demo(-1, vid_valid[0], gen_roll[0])
print(gen_roll.shape)
log_demo(-9999)

# print(vid_input.shape)
# print(n_input.shape)
# print(note_input.shape)
# print(easy_rolls.shape)
# print(vid_valid.shape)
# print(gen_roll.shape)
# print(np.max(easy_rolls))
# print(np.max(vid_input))
# print(np.max(note_input))
# print(np.max(easy_rol_valid))
# print(np.max(vid_valid_input))
# print(np.max(vid_valid))


train(easy_rolls, vid_input, note_input, EPOCHS, BATCH_SIZE)
