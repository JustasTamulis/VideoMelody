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

def make_generator_model(random_size, image_shape):

    # Input random stream
    in_lat = layers.Input(shape=(random_size,))
    n_nodes = 128
    gen = layers.Dense(n_nodes)(in_lat)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    # gen = Reshape((7, 7, 128))(gen)

    # Input image
    li_image = layers.Input(shape=image_shape)
    l_image = layers.Conv2D(9, (8, 8), strides=(2, 2), padding='same')(li_image)
    l_image = layers.Conv2D(9, (4, 4), strides=(2, 2), padding='same')(l_image)
    l_image = layers.Conv2D(3, (2, 2), padding='same')(l_image)
    # l_image = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same')(l_image)
    l_image = layers.Flatten()(l_image)
    l_image = layers.Dropout(0.01)(l_image)
    l_image = layers.Dense(128)(l_image)

    # merge image gen and label input
    merge = layers.Concatenate()([gen, l_image])

    # Output piano roll

    # output
    out_layer = layers.Dense(128, activation='hard_sigmoid')(merge)
    # define model
    model = Model([li_image, in_lat], out_layer)
    # opt = tf.keras.optimizers.Adam(1e-4)
    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def make_discriminator_model(roll_size, image_shape):
    # model = tf.keras.Sequential()
    # --------------------------------------------------
    # Input piano roll
    li_roll = layers.Input(shape=(roll_size,))
    # embedding for categorical input
    # l_roll = layers.Embedding(128, 50)(li_roll)
    # scale up to image dimensions with linear activation
    n_nodes = 30 * 30
    l_roll = layers.Dense(n_nodes)(li_roll)
    # reshape to additional channel
    # print(l_roll.shape)
    l2_roll = layers.Reshape([30, 30, 1])(l_roll)
    # print(l2_roll.shape)

    # --------------------------------------------------
    # Input image
    li_image = layers.Input(shape=image_shape)
    l_image = layers.Conv2D(9, (8, 8), strides=(2, 2), padding='same')(li_image)
    l_image = layers.Conv2D(9, (4, 4), strides=(2, 2), padding='same')(l_image)
    # l_image = layers.Conv2D(, (2, 2), strides=(2, 2), padding='same')(l_image)
    # l_image = layers.Conv2DTranspose(3, (2, 2), strides=(2, 2), padding='same')(l_image)
    # --------------------------------------------------
    # merge
    # concat label as a channel
    merge = layers.Concatenate()([l_image, l2_roll])

    # downsample
    fe = layers.Conv2D(9, (3,3), strides=(2,2), padding='same')(merge)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = layers.Conv2D(9, (3,3), strides=(2,2), padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = layers.Flatten()(fe)
    # dropout
    fe = layers.Dropout(0.4)(fe)
    # output
    out_layer = layers.Dense(1, activation='softmax')(fe)
    # define model
    model = Model([li_image, li_roll], out_layer)
    # compile model
    opt = tf.keras.optimizers.Adam(1e-4)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def make_combined_model(d_model, g_model):
    d_model.trainable = False
    g_image, g_noise = g_model.input
    g_output = g_model.output
    d_output = discriminator([g_image, g_output])
    combined_model = Model([g_image, g_noise], d_output)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    opt = tf.keras.optimizers.Adam(1e-4)
    combined_model.compile(loss=['binary_crossentropy'], metrics=['accuracy'], optimizer=opt)
    return combined_model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def train_step(img_data_batch, roll_data_batch):
    valid = np.ones((BATCH_SIZE, 1))
    valid_half = np.ones((HALF_BATCH, 1))
    half_fake = np.zeros((HALF_BATCH, 1))
    noise = np.random.normal(size=(BATCH_SIZE,100))
    half_noise = np.random.normal(size=(HALF_BATCH,100))
    # noise = tf.random.normal([BATCH_SIZE, 100])
    #  Train Discriminator
    img_data_half_batch = img_data_batch[:HALF_BATCH]
    roll_data_half_batch = roll_data_batch[:HALF_BATCH]
    d_loss_real = discriminator.train_on_batch([img_data_half_batch, roll_data_half_batch], valid_half)

    generated_roll = generator.predict([img_data_half_batch, half_noise])
    d_loss_fake = discriminator.train_on_batch([img_data_half_batch, generated_roll], half_fake)
    # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    # ---------------------
    #  Train Generator
    # ---------------------
    g_loss = combined_model.train_on_batch([img_data_batch, noise], valid)

    return d_loss_real[0], d_loss_fake[0], g_loss

def train(img_data, roll_data, epochs, batch_size):
    bat_n = int(img_data.shape[0] / batch_size)
    for epoch in range(epochs):
        ep_start = time.time()
        for b_nr in range(int(bat_n/8)):
            ba_start = time.time()
            idx = np.random.randint(0, img_data.shape[0], BATCH_SIZE)
            img_batch = img_data[idx]
            roll_batch = roll_data[idx]
            # img_batch = tf.cast(img_batch, tf.float32)
            # roll_batch = tf.cast(roll_batch, tf.float32)
            dlr, dlf, gl = train_step(img_batch, roll_batch)
            # print('d_loss_real: ' + str(dlr) + ',  d_loss_fake ' + str() g_loss")
            print('>d1=%.2f, d2=%.2f | g=%.2f   |  t=%.1f' % (dlr, dlf, gl, time.time()-ba_start))
            # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        print ('Time for epoch %d is %.3f sec'% (epoch + 1, time.time()-ep_start))

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
checkpoint_dir = folder_name + "cond-gan\\"
midi_name = 'convert_1943BossWin.mid'
avi_name = '1943.avi'
pianoroll_name = '1943.npy'
image_name = '1943_vid.npy'
result_midi_path = folder_name + 'result_1943.mid'
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

# track = Track(pianoroll=notes, program=0, is_drum=False,
#               name='loaded midi')
# multitrack = Multitrack(tracks=[track])
# pm = multitrack.to_pretty_midi()
# pm.write(result_midi_path)
# while os.path.isfile(result_midi_path) == False:
#     time.sleep(1)
# print('Converted MIDI saved')

notes = notes / 100
# maxnote = np.max(notes)
# for note in notes[:1000]:
#     cnote = []
#     for i in range(note.shape[0]):
#         if(note[i] > 0):
#             cnote.append(note[i])
#     if cnote != []:
#         print(cnote)
# print(maxnote)
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

# Check the input, PLOT IT --------------------------------------------------
# video = (video +1)/2
# print(np.min(video))
# print(np.max(video))
# fig = plt.figure()
# for i in range(1000,1500):
#     img = video[i]
#     plt.subplot(211)
#     plt.imshow(img)
#     plt.subplot(212)
#     plt.plot(notes[i])
#     plt.draw()
#     # input("<Hit Enter To Close>")
#     plt.waitforbuttonpress(0) # this will wait for indefinite time
# plt.close(fig)

# Create model --------------------------------------------------
# Create a generator and a discriminator
generator = make_generator_model(100, (IMG_HEIGHT, IMG_WIDTH, 3))
print(generator.summary())
discriminator = make_discriminator_model(pianoroll_size ,(IMG_HEIGHT, IMG_WIDTH, 3))
print(discriminator.summary())

# For the combined model we will only train the generator
combined_model = make_combined_model(discriminator, generator)
print(combined_model.summary())
cross_entropy = tf.keras.losses.BinaryCrossentropy()#from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)

# discriminator = tf.keras.models.load_model(checkpoint_dir + 'discriminator.h5')
# generator = tf.keras.models.load_model(checkpoint_dir + 'generator.h5')
# combined_model = tf.keras.models.load_model(checkpoint_dir + 'combined_model.h5')

EPOCHS = 200
BATCH_SIZE = 64
HALF_BATCH = int(BATCH_SIZE/2)

discriminator.load_weights(checkpoint_dir + 'discriminator.ckpt')
generator.load_weights(checkpoint_dir + 'generator.ckpt')
combined_model.load_weights(checkpoint_dir + 'combined_model.ckpt')

# train(image_input, roll_input, EPOCHS, BATCH_SIZE)

# discriminator.save(checkpoint_dir + 'discriminator.h5')
# generator.save(checkpoint_dir + 'generator.h5')
# combined_model.save(checkpoint_dir + 'combined_model.h5')

discriminator.save_weights(checkpoint_dir + 'discriminator.ckpt')
generator.save_weights(checkpoint_dir + 'generator.ckpt')
combined_model.save_weights(checkpoint_dir + 'combined_model.ckpt')

# discriminator = tf.keras.models.load_model(checkpoint_dir + 'discriminator.h5')
# generator = tf.keras.models.load_model(checkpoint_dir + 'generator.h5')
# combined_model = tf.keras.models.load_model(checkpoint_dir + 'combined_model.h5')

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

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
    note_threshold = 0.7
    toolbar_width = 40

    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    test_size = 10000

    # for i in range(100):
    #     if(np.sum(roll_input[i]) > 0):
    #         print(i)
    #         print(roll_input[i])

    test_input = image_input[:test_size]
    generated_roll = [roll_input[0]]



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
        sys.stdout.write("-")
        sys.stdout.flush()

    sys.stdout.write("]\n") # this ends the progress bar
    print('\nOutput:')
    print(generated_roll.shape)

    # Piano roll -> MIDI --------------------------------------------------
    track = Track(pianoroll=generated_roll, program=0, is_drum=False,
                  name='Generated midi')
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
    pm.write(result_midi_path)
    while os.path.isfile(result_midi_path) == False:
        time.sleep(1)
    print('Converted MIDI saved')

print('--------------------------------------------------')
