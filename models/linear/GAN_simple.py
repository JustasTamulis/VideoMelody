import tensorflow as tf
from keras import layers
from keras.models import Model
import numpy as np
from models.discriminator import discriminator
from models.generator import generator

# --------------------------------------------------
# ----------------------GAN-------------------------
# --------------------------------------------------
class GAN():

    def __init__(self, random_size, roll_size, image_shape,  g_optimizer, d_optimizer = None):
        if d_optimizer is None:
            d_optimizer = g_optimizer
        self.g = generator(g_optimizer, random_size, image_shape, roll_size)
        self.d = discriminator(d_optimizer, roll_size, image_shape)
        self.combined_model = self.build_model(self.g, self.d)

    def build_model(self, g, d):
        d.model.trainable = False
        g_image, g_noise = g.model.input
        g_output = g.model.output
        d_output = d.model([g_image, g_output])
        model = Model([g_image, g_noise], d_output)
        opt = g.optimizer
        model.compile(loss=['binary_crossentropy'], metrics=['acc'], optimizer=opt)
        model.name="combined_model"
        return model

    def predict(self, input_data):
         prediction = self.combined_model.predict(input_data)
         return prediction

    def train_combined(self, input_data, labels, wandb_logging_callback, verbose):
        loss = self.combined_model.fit(input_data, labels, callbacks = [wandb_logging_callback], verbose=verbose)
        return loss

    def train_step(self, img_data_batch, roll_data_batch, BATCH_SIZE, log_g, log_d, verbose):
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

        d_loss_real = self.d.train([img_data_half_batch, roll_data_half_batch], valid_half, log_d, verbose)
        generated_roll = self.g.generate([img_data_half_batch, half_noise])
        d_loss_fake = self.d.train([img_data_half_batch, generated_roll], half_fake, log_d, verbose)

        #  Train Generator
        g_loss = self.train_combined([img_data_batch, noise], valid, log_g, verbose)
        return d_loss_real, d_loss_fake, g_loss
