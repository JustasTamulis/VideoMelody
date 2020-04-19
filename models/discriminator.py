from keras import layers
from keras.models import Model
import numpy as np

# --------------------------------------------------
# ----------------DISCRIMINATOR---------------------
# --------------------------------------------------
class discriminator():

    def __init__(self, optimizer, roll_size, image_shape):
        self.optimizer = optimizer
        self.model = self.build_model(roll_size, image_shape)

    def build_model(self, roll_size, image_shape):
        # --------------------------------------------------
        # Input piano roll
        li_roll = layers.Input(shape=(roll_size,), name="Pianoroll_input")
        l_roll = layers.Dense(200,activation='relu')(li_roll)
        n_nodes = 15*15
        l_roll = layers.Dense(n_nodes,activation='relu')(l_roll)
        l2_roll = layers.Reshape([15, 15, 1])(l_roll)
        # --------------------------------------------------
        # Input image
        li_image = layers.Input(shape=image_shape, name="Image_input")
        l_image = layers.Conv2D(27, (4, 4), padding='same',activation='relu')(li_image)
        l_image = layers.MaxPooling2D((4, 4))(l_image)
        l_image = layers.Conv2D(9, (4, 4),  padding='same',activation='relu')(l_image)
        l_image = layers.MaxPooling2D((2, 2))(l_image)
        # --------------------------------------------------
        # merge
        merge = layers.Concatenate()([l_image, l2_roll])
        # --------------------------------------------------
        # downsample
        fe = layers.Conv2D(3, (3,3), padding='same', activation='relu')(merge)
        fe = layers.LeakyReLU(alpha=0.2)(fe)
        fe = layers.Flatten()(fe)
        # output
        out_layer = layers.Dense(1, activation='sigmoid', name="Discriminator_decision")(fe)
        # define model
        model = Model([li_image, li_roll], out_layer)
        # compile model
        opt = self.optimizer
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
        model.name="Discriminator"
        return model

    def predict(self, input_data):
         prediction = self.model.predict(input_data)
         return prediction

    def train(self, input_data, labels, wandb_logging_callback, verbose):
        loss = self.model.fit(input_data, labels, callbacks = [wandb_logging_callback], verbose=verbose)
        return loss
