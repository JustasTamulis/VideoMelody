import os
import random
import time
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pygame
import pretty_midi
from moviepy.editor import *
from pypianoroll import Multitrack, Track
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.utils import plot_model
import tensorflow as tf
from models import *
from keras.callbacks import LambdaCallback
import wandb
from wandb.keras import WandbCallback
from keras.models import load_model
import argparse

wandb.init(project='videomelody')
config = wandb.config

# --------------------------------------------------
# ---------------HYPER-PARAMETERS-------------------
# --------------------------------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy()#from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(lr = 1e-2)#, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lr = 1e-2)#, beta_1=0.5)

EPOCHS = 50
BATCH_SIZE = 128
HALF_BATCH = int(BATCH_SIZE/2)

config.epochs = 3
config.discriminator_epochs = 1
config.discriminator_examples = 10000
config.generator_epochs = 12
config.generator_examples = 10000
config.batch_size = 256
config.half_batch_size = 256
config.image_shape = (120, 120, 3)
# --------------------------------------------------


def log_generator(epoch, logs):
    wandb.log({'generator_loss': logs['loss'],
                     'generator_acc': logs['acc'],
                     'discriminator_loss': 0.0,
                     'discriminator_acc': (1-logs['acc'])/2.0+0.5})

def log_discriminator(epoch, logs):
    wandb.log({
            'generator_loss': 0.0,
            'generator_acc': (1.0-logs['acc'])*2.0,
            'discriminator_loss': logs['loss'],
            'discriminator_acc': logs['acc']})

def train(img_data, roll_data, epochs, batch_size):
    bat_n = int(img_data.shape[0] / batch_size)

    wandb_logging_g = LambdaCallback(on_epoch_end=log_generator)
    wandb_logging_d = LambdaCallback(on_epoch_end=log_discriminator)

    for epoch in range(epochs):
        for b_nr in range(int(bat_n)):

            idx = np.random.randint(0, img_data.shape[0], BATCH_SIZE)
            img_batch = img_data[idx]
            roll_batch = roll_data[idx]

            gan.train_step(img_batch, roll_batch, BATCH_SIZE, wandb_logging_g, wandb_logging_d, verbose)

        # print('Time for epoch %d is %.3f sec'% (epoch + 1, time.time()-ep_start))



folder_name = "F:\\CompSci\\project\\Data\\Piano midi\\"
checkpoint_dir = folder_name + "checkpoints3\\"
flat_midi_name = 'flat_midi.mid'
midi_name = 'faure-dolly-suite-1-berceuse.mid'
avi_name = 'flat_faure.avi'
pianoroll_name = 'roll_ndarray.npy'
image_name = 'image_ndarray.npy'
result_midi_path = folder_name + 'result_faure.mid'
midi_path = folder_name + midi_name
avi_path = folder_name + avi_name
piano_path = folder_name + pianoroll_name
image_path = folder_name + image_name
history_path = folder_name + 'history.npy'
changed_input_path = folder_name + 'changed_input.npy'

# LOAD INPPUT TO NETWORK --------------------------------------------------
notes = np.load(piano_path)
pianoroll_size = 128
random_size=100
roll_count = notes.shape[0]
IMG_HEIGHT = 120
IMG_WIDTH = 120
roll_input = notes
roll_input = roll_input.astype('float32')
image_input = np.load(changed_input_path)

# Create model --------------------------------------------------

# parser = argparse.ArgumentParser(description='Wandb example GAN')
# parser.add_argument('--disc', type=str, default=None, metavar='N',
#                     help='link to discriminator model file')
# parser.add_argument('--gen', type=str, default=None, metavar='N',
#                     help='link to generator model file')
# args = parser.parse_args()

gan = GAN(random_size, pianoroll_size ,(IMG_HEIGHT, IMG_WIDTH, 3), generator_optimizer, discriminator_optimizer)
verbose = 0
train(image_input, roll_input, EPOCHS, BATCH_SIZE)


test = False
if test:
    # Test generator and discriminator alone
    img_test = np.reshape(image_input[0], (1, IMG_WIDTH, IMG_HEIGHT, 3)).astype('float32')
    roll_test = np.reshape(roll_input[0], (1, pianoroll_size)).astype('float32')
    # print(img_test.shape)
    # print(roll_test.shape)
    prediction = discriminator.predict([img_test, roll_test])
    print('PREDICTION: ' + str(prediction))
    seed = np.random.normal(size=(1,100))
    # seed = np.reshape(seed, (1, 100)).astype('float32')
    generated_roll = generator.predict([img_test, seed])
    print(generated_roll)

test = False
if test:
    # OUTPUT FROM NETWORK
    print('\nTESTING AND CREATING')
    note_threshold = 0.3
    toolbar_width = 40

    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    test_size = image_input.shape[0]
    test_input = image_input[:test_size]
    generated_roll = [roll_input[0]]
    i = 0
    for t_in in test_input:

        noise = np.random.normal(size=(1,100))
        img_test = np.reshape(t_in, (1, IMG_WIDTH, IMG_HEIGHT, 3)).astype('float32')
        gen_roll = generator.predict([img_test, noise])
        # print(gen_roll)
        trans = np.array(gen_roll[0, :])
        trans = np.where(trans < note_threshold, 0, 1)
        # print(trans.shape)
        # print(generated_roll.shape)
        generated_roll = np.append(generated_roll, [trans * 100 ], axis = 0)
        i = i + 1
        if(i > test_size/toolbar_width):
            i = 0
            sys.stdout.write("-")
            sys.stdout.flush()

    sys.stdout.write("]\n") # this ends the progress bar
    print('\nOutput:')
    print(generated_roll.shape)


    fig = plt.figure()
    for i in range(0,100):
        fig.clf()
        img = image_input[i]
        plt.subplot(211)
        plt.imshow(img)
        plt.subplot(212)
        plt.plot(generated_roll[i])
        plt.draw()
        # input("<Hit Enter To Close>")
        plt.waitforbuttonpress(0) # this will wait for indefinite time
    plt.close(fig)

    # Piano roll -> MIDI --------------------------------------------------
    average_tempo = 64.3
    track = Track(pianoroll=generated_roll, program=0, is_drum=False,
                  name='Generated midi')
    multitrack = Multitrack(tracks=[track], beat_resolution=20, tempo = average_tempo)

    fig, axs = multitrack.plot()
    plt.show()

    pm = multitrack.to_pretty_midi()
    print('midi from OUTPUT  end time ' + str(pm.get_end_time()))

    # MIDI play --------------------------------------------------
    memFile = BytesIO()
    pm.write(memFile)
    memFile.seek(0)
    playMidiFile(memFile)

print('--------------------------------------------------')
