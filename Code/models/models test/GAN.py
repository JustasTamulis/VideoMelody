import tensorflow as tf
from keras import layers
from keras.models import Model
import numpy as np
from models.discriminator import discriminator
from models.generator import generator
import os
from scipy.special import softmax
from keras import backend as K
# --------------------------------------------------
# ----------------------GAN-------------------------
# --------------------------------------------------
class GAN():

    def __init__(self, random_size, roll_size, image_shape, sequence_length, g_optimizer = None, d_optimizer = None):
        if g_optimizer is None:
            g_optimizer = tf.keras.optimizers.Adam(lr = 1e-4, beta_1=0.5)
        if d_optimizer is None:
            d_optimizer = g_optimizer
        self.random_size = random_size
        self.g = generator(g_optimizer, random_size, image_shape, roll_size, sequence_length)
        self.d = discriminator(d_optimizer, roll_size, image_shape, sequence_length)
        # self.d.summary()
        # print(self.g.model.summary())
        self.combined_model = self.build_model(self.g, self.d)



    def build_model(self, g, d):
        d.model.trainable = False
        g_image, g_noise, g_extra = g.model.input
        g_output = g.model.output
        d_output = d.model([g_image, g_output, g_extra])
        model = Model([g_image, g_noise, g_extra], d_output)
        opt = g.optimizer

        def custom_loss(y_true, y_pred):
            return tf.compat.v1.losses.sigmoid_cross_entropy(y_true, y_pred, label_smoothing=0.01)

        model.compile(loss=custom_loss, metrics=['acc'], optimizer=opt)
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

    def train_step(self, note_batch, img_data_batch, roll_data_batch, BATCH_SIZE, log_g, log_d, verbose):
        HALF_BATCH = int(BATCH_SIZE/2)

        # Create noise
        valid = np.random.normal(1, 0.1, (BATCH_SIZE,1))
        valid_half = np.random.normal(1, 0.1, (HALF_BATCH,1))
        half_fake = np.absolute(np.random.normal(0, 0.1, (HALF_BATCH,1)))
        fake = np.absolute(np.random.normal(0, 0.1, (BATCH_SIZE,1)))
        noise = np.random.normal(size=(BATCH_SIZE,self.random_size))
        half_noise = np.random.normal(size=(HALF_BATCH,self.random_size))

        # Prepare inputs
        img_data_half_batch = img_data_batch[:HALF_BATCH]
        roll_data_half_batch = roll_data_batch[:HALF_BATCH]
        rand_for_roll = np.absolute(np.random.normal(0, 0.01, (roll_data_half_batch.shape)))
        roll_data_half_batch = roll_data_half_batch + rand_for_roll
        roll_data_half_batch = softmax(roll_data_half_batch, axis = -1)
        rand_for_notes = np.absolute(np.random.normal(0, 0.01, (note_batch.shape)))
        note_batch = note_batch + rand_for_notes
        note_batch = softmax(note_batch, axis = -1)
        note_half_batch = note_batch[:HALF_BATCH]

        # Generate fake roll
        generated_roll = self.g.generate([img_data_half_batch, half_noise, note_half_batch])
        diff_pitches = set()
        for i in range(len(generated_roll)):
            nr = np.argmax(generated_roll[i])
            diff_pitches.add(nr)
        pitch_count = len(diff_pitches)

        #  Train Discriminator if loss is more than 0.6
        d_loss_real = self.d.evaluate([img_data_half_batch, roll_data_half_batch, note_half_batch], valid_half, verbose, HALF_BATCH)
        d_loss_fake = self.d.evaluate([img_data_half_batch, generated_roll, note_half_batch], half_fake, verbose, HALF_BATCH)

        if d_loss_real[0] > 0.6:
            d_real_hist = self.d.train([img_data_half_batch, roll_data_half_batch, note_half_batch], valid_half, log_d, verbose, HALF_BATCH)
            d_loss_real = d_real_hist.history['loss'][0]

        if d_loss_fake[0] > 0.6:
            d_fake_hist = self.d.train([img_data_half_batch, generated_roll, note_half_batch], half_fake, log_d, verbose, HALF_BATCH)
            d_loss_fake = d_fake_hist.history['loss'][0]

        #  Train Generator
        g_loss = self.train_combined([img_data_batch, noise, note_batch], fake, log_g, verbose, BATCH_SIZE)

        return d_loss_real, d_loss_fake, g_loss.history['loss'][0], diff_pitches

    def summary(self):
         self.d.summary()
         print(self.combined_model.summary())

    def get_models(self):
        return self.combined_model, self.g.get_model()

    def load_weights(self, gen, dis):

        self.combined_model.load_weights(gen)
        self.g.get_model().load_weights(dis)
