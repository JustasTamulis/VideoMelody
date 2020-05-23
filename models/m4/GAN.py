import tensorflow as tf
from keras import layers
from keras.models import Model
import numpy as np
from models.m4.discriminator import discriminator
from models.m4.generator import generator
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

        model.compile(loss=custom_loss, metrics=['binary_accuracy'], optimizer=opt)
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

    def train_step(self, cond_roll_batch, cond_note_batch, img_data_batch, roll_data_batch, BATCH_SIZE, log_g, log_d, verbose):
        HALF_BATCH = int(BATCH_SIZE/2)

        # Numpy labels with noise
        valid = np.random.normal(0.9, 0.1, (BATCH_SIZE,1))
        valid_half = np.random.normal(0.9, 0.1, (HALF_BATCH,1))
        # half_fake = np.absolute(np.random.normal(0, 0.1, (HALF_BATCH,1)))
        # fake = np.absolute(np.random.normal(0, 0.1, (BATCH_SIZE,1)))

        # Numpy labels Without label smoothing
        half_fake =  np.zeros((HALF_BATCH, 1))
        fake =  np.zeros((BATCH_SIZE, 1))
        # valid_half =  np.ones((HALF_BATCH, 1))
        # valid =  np.ones((BATCH_SIZE, 1))

        # # Tensor labels
        # half_fake =  K.zeros((HALF_BATCH, 1))
        # fake =  K.zeros((BATCH_SIZE, 1))
        # valid_half =  K.ones((HALF_BATCH, 1))
        # valid =  K.ones((BATCH_SIZE, 1))

        noise = np.random.normal(size=(BATCH_SIZE,self.random_size))
        half_noise = np.random.normal(size=(HALF_BATCH,self.random_size))

        # Prepare inputs
        img_data_half_batch = img_data_batch[:HALF_BATCH]
        roll_data_half_batch = roll_data_batch[:HALF_BATCH]
        rand_for_roll = np.absolute(np.random.normal(0, 0.01, (roll_data_half_batch.shape)))
        roll_data_half_batch = roll_data_half_batch + rand_for_roll
        roll_data_half_batch = softmax(roll_data_half_batch, axis = -1)
        rand_for_notes = np.absolute(np.random.normal(0, 0.01, (cond_roll_batch.shape)))
        cond_roll_batch = cond_roll_batch + rand_for_notes
        cond_roll_batch = softmax(cond_roll_batch, axis = -1)
        note_half_batch = cond_roll_batch[:HALF_BATCH]

        # Generate fake roll
        generated_roll = self.g.generate([img_data_half_batch, half_noise, note_half_batch])
        diff_pitches = set()
        prev_pitch = -1
        pitch_dur = 0
        dur = list()
        for i in range(len(generated_roll)):
            for j in range(len(generated_roll[i])):
                nr = np.argmax(generated_roll[i][j])
                diff_pitches.add(nr)
                if nr != prev_pitch:
                    pitch_dur = pitch_dur + 1
                else:
                    dur.append(pitch_dur)
                    pitch_dur = 0
        if pitch_dur != 0:
            dur.append(pitch_dur)
        pitch_freq = sum(dur) / len(dur)

        #  Train Discriminator if loss is more than 0.05
        [d_loss_real, d_acc_real] = self.d.evaluate([img_data_half_batch, roll_data_half_batch, note_half_batch], valid_half, verbose, HALF_BATCH)
        [d_loss_fake, d_acc_fake] = self.d.evaluate([img_data_half_batch, generated_roll, note_half_batch], half_fake, verbose, HALF_BATCH)

        if d_loss_real > 0.05:
            d_real_hist = self.d.train([img_data_half_batch, roll_data_half_batch, note_half_batch], valid_half, log_d, verbose, HALF_BATCH)
            d_loss_real = d_real_hist.history['loss'][0]
            d_acc_real = d_real_hist.history['binary_accuracy'][0]

        if d_loss_fake > 0.05:
            d_fake_hist = self.d.train([img_data_half_batch, generated_roll, note_half_batch], half_fake, log_d, verbose, HALF_BATCH)
            d_loss_fake = d_fake_hist.history['loss'][0]
            d_acc_fake = d_fake_hist.history['binary_accuracy'][0]

        #  Train Generator
        g_loss = self.train_combined([img_data_batch, noise, cond_roll_batch], fake, log_g, verbose, BATCH_SIZE)

        return d_loss_real, d_acc_real, d_loss_fake, d_acc_fake, g_loss.history['loss'][0], g_loss.history['binary_accuracy'][0], diff_pitches, pitch_freq


    def summary(self):
         self.d.summary()
         print(self.combined_model.summary())

    def get_models(self):
        return self.combined_model, self.g.get_model()

    def load_weights(self, gen, dis):

        self.combined_model.load_weights(gen)
        self.g.get_model().load_weights(dis)
