# LSTM GAN for piano rolls



import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pretty_midi
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from moviepy.editor import *
import numpy as np
from pypianoroll import Multitrack, Track
from io import BytesIO
import random
import time
import cv2
print('---------------------------------')

def playMidiFile(midi_stream):
    pygame.mixer.music.load(midi_stream)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    return True

# define the LSTM model
def make_generator_model(input_length, input_size):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(256, return_sequences=True, input_shape=(input_length, input_size)))
    model.add(layers.Dropout(0.2))
    # model.add(layers.LSTM(256, return_sequences=True))
    # model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(128))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def make_discriminator_model(input_length, input_size):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(256, return_sequences=True, input_shape=(input_length, input_size)))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(128))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='softmax'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def train_step(notes_batch):
    # noise = tf.random.normal([BATCH_SIZE, pianoroll_size])
    # noise = np.random.normal(0, 1, (BATCH_SIZE, self.latent_dim))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_note = generator(notes_batch, training=True)

      # Append generate note to the sequence.
      # print(tf.shape(generated_note))
      generated_note = tf.reshape(generated_note, [64, 1, 128])
      # print(tf.shape(generated_note))
      generated_notes = tf.concat([notes_batch, generated_note], 1)

      real_output = discriminator(notes_batch, training=True)
      fake_output = discriminator(generated_notes, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        idx = np.random.randint(0, dataset.shape[0], BATCH_SIZE)
        real_seqs = dataset[idx]
        # print(type(real_seqs))
        # print(real_seqs.shape)
        real_seqs = tf.cast(real_seqs, tf.float32)
        train_step(real_seqs)
            # print(real_seqs.shape)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Save the model every after last epoch
    # if (epoch + 1) % 20 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)


folder_name = "F:\\CompSci\\project\\MIDI\\ungrouped midi\\"
midi_name = 'convert_1943BossWin.mid'
avi_name = '1943.avi'
pianoroll_name = '1943.npy'
midi_path = folder_name + midi_name
avi_path = folder_name + avi_name
piano_path = folder_name + pianoroll_name

# SOUND
# Read PIANO ROLL --------------------------------------------------
notes = np.load(piano_path)
print('Input piano roll shape: ' + str(notes.shape))
pianoroll_size = 128

# CREATE INPPUT TO NETWORK --------------------------------------------------

sequence_length = 100
network_input = []
network_output = []
# create input sequences and the corresponding outputs
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append(sequence_in)
    network_output.append(sequence_out)
n_patterns = len(network_input)
# reshape the input into a format compatible with LSTM layers
network_input = np.reshape(network_input, (n_patterns, sequence_length, pianoroll_size))
# network_output = np.reshape(network_output, (n_patterns, sequence_length, pianoroll_size))
network_output = np.array(network_output)
print('Network input shape: ' + str(network_input.shape))
# print('Network expected output shape: ' + str(len(network_output)))
# print('Network expected output type: ' + str(type(network_output)))
print('expected output shape: ' + str(network_output.shape))
# BUFFER_SIZE = video.shape[0]
# train_dataset = tf.data.Dataset.from_tensor_slices(network_input).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# Create model --------------------------------------------------
# Create a generator and a discriminator
generator = make_generator_model(sequence_length, pianoroll_size)
# print(generator.summary())

discriminator = make_discriminator_model(sequence_length, pianoroll_size)
# print(discriminator.summary())

cross_entropy = tf.keras.losses.BinaryCrossentropy()#from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint_dir = folder_name + 'frame-gan\\'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 500
BATCH_SIZE = 64


# train(network_input, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

test = True
if test:
    # OUTPUT FROM NETWORK
    print('\nTESTING AND CREATING')
    note_threshold = 0.9

    # Create one piano roll frame for testing
    starting_input = network_input[0]

    # print(np.sum(starting_input, axis=1).shape)
    generated_roll = starting_input
    current_roll = starting_input

    toolbar_width = 10

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
