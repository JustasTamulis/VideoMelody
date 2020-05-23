# from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
import random
tf.enable_eager_execution()


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(5*5*4*3*3, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((15, 20, 3)))

    model.add(layers.Conv2DTranspose(27, (2, 2), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(18, (5, 5), strides=(3, 3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(4, 4), padding='same', use_bias=False, activation='tanh'))

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(27, (10, 10), strides=(4, 4), padding='same',
                                     input_shape=[360, 480, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(9, (5, 5), strides=(8, 8), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        # print("epoch  " + str(epoch))
        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 5 epochs
        if (epoch + 1) % 20 == 0:
          checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,epochs, seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  # fig = plt.figure(figsize=(1,1))
  # img = None
  # for i in range(predictions.shape[0]):
  #     plt.subplot(1, 1, i+1)
  img = np.array((predictions[0, :, :, :] *127.5 + 127.5) / 255.0)
  plt.imshow(img)
  plt.axis('off')

  print("maximum " + str(np.amax(img)) + ", minimum " + str(np.amin(img)))
  plt.savefig('./test3/image_one_{:04d}.png'.format(epoch))


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


video_name = 'HarderBetterFasterStronger.mp4'
video = getVideo(video_name)

FRAMES = video.shape[0]
IMG_HEIGHT = 360
IMG_WIDTH = 480
print(FRAMES)

video = video.reshape(FRAMES, IMG_HEIGHT, IMG_WIDTH, 3).astype('float32')
video = video[:1024,:,:,:]
video = (video - 127.5) / 127.5
BUFFER_SIZE = FRAMES
BATCH_SIZE = 64
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(video).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# print(train_dataset)

generator = make_generator_model()
discriminator = make_discriminator_model()
print(generator.summary())
print(discriminator.summary())
# noise = tf.random.normal([1, 100])
# generated_image = generator(noise, training=False)
# img = np.array(((generated_image[0, :, :, :] *127.5 + 127.5) / 255.0))
# print("maximum " + str(np.amax(img)))
# print("minimum " + str(np.amin(img)))
# plt.imshow(img)
# plt.savefig('image_good.png')
# plt.show()
# #
# #
# # print(type(generated_image))
# # print(generated_image.shape)
# gen = (generated_image[0, :, :, :] + 1) / 2
# print(gen)
# plt.imshow((gen))
# plt.show()
#
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './test3/training_checkpoints2'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 1
seed = tf.random.normal([num_examples_to_generate, noise_dim])

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



img = None
noise = tf.random.normal([1, 100])
for i in range(1000):
    noise = tf.random.normal([1, 100]) * 0.1 + noise
    # for x in range(100):
    #     if random.randint(0, 10) == 10:
    #         noise[x] = random.uniform(0, 1)
    generated_image = generator(noise, training=False)
    imag = np.array(((generated_image[0, :, :, :] *127.5 + 127.5) / 255.0))
    # print("maximum " + str(np.amax(imag)))
    # print("minimum " + str(np.amin(imag)))

    if img is None:
        img = plt.imshow(imag)
    else:
        img.set_data(imag)
    plt.pause(.1)
    plt.draw()



# train(train_dataset, EPOCHS)



# print(generated_image[0, :, :, 0])
# print(generated_image[0, :, :, 0])
# print(generated_image.type)
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()
# input("Press Enter to continue...")
