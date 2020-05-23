# LSTM GAN for piano rolls



import os
import random
import time
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pretty_midi
from moviepy.editor import *
from pypianoroll import Multitrack, Track
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import cv2

import tensorflow as tf
from keras import layers
from keras.models import Model
# from tensorflow.keras import layers
print('---------------------------------')

def playMidiFile(midi_stream):
    pygame.mixer.music.load(midi_stream)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    return True

# define the LSTM model
def make_generator_model(random_size, image_shape):

    # Input random stream
    in_lat = layers.Input(shape=(random_size,))
    # foundation for 7x7 image
    n_nodes = 120
    gen = layers.Dense(n_nodes)(in_lat)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    # gen = Reshape((7, 7, 128))(gen)

    # Input image
    li_image = layers.Input(shape=image_shape)
    l_image = layers.Conv2D(9, (2, 2), strides=(2, 2), padding='same')(li_image)
    l_image = layers.Conv2DTranspose(3, (2, 2), strides=(2, 2), padding='same')(l_image)
    l_image = layers.Flatten()(l_image)
    l_image = layers.Dense(120)(l_image)

    # merge image gen and label input
    merge = layers.Concatenate()([gen, l_image])

    # Output piano roll

    # output
    out_layer = layers.Dense(128, activation='softmax')(merge)
    # define model
    model = Model([li_image, in_lat], out_layer)
    opt = tf.keras.optimizers.Adam(1e-4)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def make_discriminator_model(roll_size, image_shape):
    # model = tf.keras.Sequential()
    # --------------------------------------------------
    # Input piano roll
    li_roll = layers.Input(shape=(roll_size,))
    # embedding for categorical input
    # l_roll = layers.Embedding(128, 50)(li_roll)
    # scale up to image dimensions with linear activation
    n_nodes = image_shape[0] * image_shape[1] * image_shape[2]
    l_roll = layers.Dense(n_nodes)(li_roll)
    # reshape to additional channel
    # print(l_roll.shape)
    l2_roll = layers.Reshape([image_shape[0], image_shape[1], image_shape[2]])(l_roll)
    # print(l2_roll.shape)

    # --------------------------------------------------
    # Input image
    li_image = layers.Input(shape=image_shape)
    l_image = layers.Conv2D(9, (2, 2), strides=(2, 2), padding='same')(li_image)
    l_image = layers.Conv2DTranspose(3, (2, 2), strides=(2, 2), padding='same')(l_image)


    # --------------------------------------------------
    # merge

    # concat label as a channel
    merge = layers.Concatenate()([l_image, l2_roll])

    # Output True / False

    # downsample
    fe = layers.Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = layers.Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = layers.Flatten()(fe)
    # dropout
    fe = layers.Dropout(0.4)(fe)
    # output
    out_layer = layers.Dense(1, activation='sigmoid')(fe)
    # define model
    model = Model([li_image, li_roll], out_layer)
    # compile model
    opt = tf.keras.optimizers.Adam(1e-4)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def train_step(img_data, roll_data):
    valid = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))
    noise = np.random.normal(size=(BATCH_SIZE,100))
    # noise = tf.random.normal([BATCH_SIZE, 100])

    # ---------------------
    #  Train Discriminator
    # ---------------------
    generated_roll = generator.predict([img_data, noise])

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch([img_data, roll_data], valid)
    d_loss_fake = discriminator.train_on_batch([img_data, generated_roll], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    # ---------------------
    #  Train Generator
    # ---------------------

    # Condition on labels
    # sampled_labels = np.random.randint(0, 10, BATCH_SIZE).reshape(-1, 1)

    # Train the generator
    g_loss = combined_model.train_on_batch([img_data, noise], valid)
    #
    # real_output = discriminator([img_data, roll_data])#, training=True)
    # fake_output = discriminator([img_data, generated_roll])#, training=True)
    #
    # gen_loss = generator_loss(fake_output)
    # disc_loss = discriminator_loss(real_output, fake_output)



def train(img_data, roll_data, epochs):
    for epoch in range(epochs):
        start = time.time()
        idx = np.random.randint(0, img_data.shape[0], BATCH_SIZE)
        img_batch = img_data[idx]
        roll_batch = roll_data[idx]
        # img_batch = tf.cast(img_batch, tf.float32)
        # roll_batch = tf.cast(roll_batch, tf.float32)
        train_step(img_batch, roll_batch)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Save the model every after last epoch
    # if (epoch + 1) % 20 == 0:
    # checkpoint.save(file_prefix = checkpoint_prefix)

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
    return buf


folder_name = "F:\\CompSci\\project\\MIDI\\ungrouped midi\\"
midi_name = 'convert_1943BossWin.mid'
avi_name = '1943.avi'
pianoroll_name = '1943.npy'
image_name = '1943_vid.npy'
midi_path = folder_name + midi_name
avi_path = folder_name + avi_name
piano_path = folder_name + pianoroll_name
image_path = folder_name + image_name

print('Input raw data')
# SOUND
# Read PIANO ROLL --------------------------------------------------
notes = np.load(piano_path)
print('piano roll shape: ' + str(notes.shape))
pianoroll_size = 128
roll_count = notes.shape[0]

# VIDEO
# Read avi file and construct --------------------------------------------------

# video = getVideo(avi_path)
video = np.load(image_path)

FRAMES = video.shape[0]
IMG_HEIGHT = 120
IMG_WIDTH = 120

# video = video.reshape(FRAMES, IMG_HEIGHT, IMG_WIDTH, 3).astype('float32')
# video = (video - 127.5) / 127.5

# np.save(image_path, video)

print('video shape: ' + str(video.shape))
print()
# CREATE INPPUT TO NETWORK --------------------------------------------------

roll_input = notes
image_input = []
for i in range(roll_count):
    idx = min(int(i / 4.7), FRAMES-1)
    image_in = video[idx ,:,:,:]
    image_input.append(image_in)
print("Input to the network")
print("Piano roll shape: " + str(roll_input.shape))

image_input = np.reshape(image_input, (roll_count, IMG_HEIGHT, IMG_WIDTH, 3))
print("Images shape: " + str(image_input.shape))

# Create model --------------------------------------------------
# Create a generator and a discriminator
generator = make_generator_model(100, (IMG_HEIGHT, IMG_WIDTH, 3))
# print(generator.summary())

discriminator = make_discriminator_model(pianoroll_size ,(IMG_HEIGHT, IMG_WIDTH, 3))
# print(discriminator.summary())

# For the combined model we will only train the generator
noise = layers.Input(shape=(100,))
img = layers.Input(shape=(120,120,3,))
roll = generator([img, noise])


discriminator.trainable = False
valid = discriminator([img, roll])
# The combined model  (stacked generator and discriminator)
# Trains generator to fool discriminator
combined_model = Model([img, noise], valid)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
combined_model.compile(loss=['binary_crossentropy'], optimizer=generator_optimizer)

cross_entropy = tf.keras.losses.BinaryCrossentropy()#from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint_dir = folder_name + 'cond-gan\\'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)

EPOCHS = 2
BATCH_SIZE = 64


train(image_input, roll_input, EPOCHS)
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

test = False
if test:
    # Test generator and discriminator alone
    img_test = np.reshape(image_input[0], (1, IMG_WIDTH, IMG_HEIGHT, 3)).astype('float32')
    roll_test = np.reshape(roll_input[0], (1, pianoroll_size)).astype('float32')
    print(img_test.shape)
    print(roll_test.shape)
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
    note_threshold = 0.9

    # Create one piano roll frame for testing
    starting_input = network_input[0]

    # print(np.sum(starting_input, axis=1).shape)
    generated_roll = starting_input
    current_roll = starting_input

    toolbar_width = 40

    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    for t in range(100):
        x = np.reshape(current_roll, (1, sequence_length, pianoroll_size)).astype('float32')
        prediction = generator(x, training=False)
        trans = np.array(prediction[0, :])
        trans = np.where(trans < note_threshold, 0, 1)

        generated_roll = np.append(generated_roll, [trans], axis = 0)
        current_roll = np.append(current_roll, [trans], axis = 0)
        current_roll = current_roll[1:]
        # print(current_roll.shape)

        # update the bar
        sys.stdout.write("-")
        sys.stdout.flush()
    sys.stdout.write("]\n") # this ends the progress bar
    print('\nOutput:')
    print(generated_roll.shape)

    # Piano roll -> MIDI --------------------------------------------------
    track = Track(pianoroll=generated_roll, program=0, is_drum=False,
                  name=midi_name)
    multitrack = Multitrack(tracks=[track])

    fig, axs = multitrack.plot()
    plt.show()

    pm = multitrack.to_pretty_midi()
    print('midi from OUTPUT  end time ' + str(pm.get_end_time()))

    # MIDI play --------------------------------------------------
    memFile = BytesIO()
    pm.write(memFile)
    memFile.seek(0)
    playMidiFile(memFile)

    # print(generated_roll)
    # print(np.sum(generated_roll, axis=1))
    # --------------------------------------------------/
    # ----------------------OLD-------------------------/
    # --------------------------------------------------/
    #

    # MIDI save --------------------------------------------------
    # pm.write(converted_midi_path)
    # while os.path.isfile(converted_midi_path) == False:
    #     time.sleep(1)
    # print('Converted MIDI saved')

    print('--------------------------------------------------')
