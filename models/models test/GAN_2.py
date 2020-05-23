import tensorflow as tf
from keras import layers
from keras.models import Model
import numpy as np
from models.discriminator import discriminator
from models.generator import generator
import os
# --------------------------------------------------
# ----------------------GAN-------------------------
# --------------------------------------------------
class GAN():

    def __init__(self, random_size, roll_size, image_shape, sequence_length, g_optimizer, d_optimizer = None):
        if d_optimizer is None:
            d_optimizer = g_optimizer
        self.g = generator(g_optimizer, random_size, image_shape, roll_size, sequence_length)
        self.d = discriminator(d_optimizer, roll_size, image_shape, sequence_length)
        self.combined_model = self.build_model(self.g, self.d)

    def build_model(self, g, d):
        d.model.trainable = False
        g_image, g_noise, g_roll = g.model.input
        g_output = g.model.output
        d_output = d.model([g_image, g_output])
        model = Model([g_image, g_noise, g_roll], d_output)
        opt = g.optimizer
        model.compile(loss=['binary_crossentropy'], metrics=['acc'], optimizer=opt)
        model.name="combined_model"
        return model

    def predict(self, input_data):
         prediction = self.combined_model.predict(input_data)
         return prediction

    def generate(self, input):
        return self.g.generate(input)

    def train_combined(self, input_data, labels, wandb_logging_callback, verbose, batch_size):
        loss = self.combined_model.fit(input_data, labels, batch_size, callbacks = [], verbose=verbose)
        return loss

    def train_step(self, easy_rolls, img_data_batch, roll_data_batch, BATCH_SIZE, log_g, log_d, verbose):
        HALF_BATCH = int(BATCH_SIZE/2)

        # Create output and noise
        valid = np.ones((BATCH_SIZE, 1))
        valid_half = np.ones((HALF_BATCH, 1))
        half_fake = np.zeros((HALF_BATCH, 1))
        fake = np.zeros((BATCH_SIZE, 1))
        noise = np.random.normal(size=(BATCH_SIZE,100))
        half_noise = np.random.normal(size=(HALF_BATCH,100))

        #  Train Discriminator
        img_data_half_batch = img_data_batch[:HALF_BATCH]
        roll_data_half_batch = roll_data_batch[:HALF_BATCH]
        half_easy_rolls = easy_rolls[:HALF_BATCH]

        d_loss_real = self.d.train([img_data_half_batch, roll_data_half_batch], valid_half, log_d, verbose, HALF_BATCH)
        generated_roll = self.g.generate([img_data_half_batch, half_noise, half_easy_rolls])

        diff_pitches = set()
        for i in range(len(generated_roll)):
            nr = np.argmax(generated_roll[i])
            diff_pitches.add(nr)
        pitch_count = len(diff_pitches)

        d_loss_fake = self.d.train([img_data_half_batch, generated_roll], half_fake, log_d, verbose, HALF_BATCH)

        #  Train Generator
        g_loss = self.train_combined([img_data_batch, noise, easy_rolls], valid, log_g, verbose, BATCH_SIZE)
        # g_loss = self.train_combined([img_data_batch, noise, easy_rolls], valid, log_g, verbose, BATCH_SIZE)

        return d_loss_real, d_loss_fake, g_loss, pitch_count

    def summary(self):
         self.d.summary()
         print(self.combined_model.summary())

    def get_models(self):
        return self.combined_model, self.g.get_model()
