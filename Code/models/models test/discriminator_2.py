from keras import layers
from keras.models import Model
import numpy as np
# --------------------------------------------------
# (sequence_length, roll_size) + (sequence_length, video_vec_size) -> T/F
# --------------------------------------------------
# ----------------DISCRIMINATOR---------------------
# --------------------------------------------------
class discriminator():

    def __init__(self, optimizer, roll_size, image_shape, sequence_length):
        self.optimizer = optimizer
        self.model = self.build_model(roll_size, image_shape, sequence_length)

    def build_model(self, roll_size, image_shape, sequence_length):
        # --------------------------------------------------
        # Input piano roll
        li_roll = layers.Input(shape=(roll_size,), name="Pianoroll_input")
        l_roll = layers.Dense(100,activation='relu')(li_roll)
        l_roll = layers.Dense(100,activation='relu')(l_roll)
        # --------------------------------------------------
        # Input image
        li_image = layers.Input(shape=(sequence_length, image_shape), name="Image_input")
        l_image = layers.LSTM(256, input_shape=(sequence_length, image_shape),return_sequences=True)(li_image)
        l_image = layers.Bidirectional(layers.LSTM(128))(l_image)
        l_image = layers.BatchNormalization()(l_image)
        l_image = layers.Dense(100)(l_image)
        l_image = layers.LeakyReLU(alpha=0.1)(l_image)
        l_image = layers.Dropout(0.05)(l_image)
        l_image = layers.Dense(100, activation='relu')(l_image)
        l_image = layers.BatchNormalization()(l_image)
        # --------------------------------------------------
        # merge
        merge = layers.Concatenate()([l_image, l_roll])
        l_comb = layers.Dense(100)(merge)
        fe = layers.LeakyReLU(alpha=0.2)(l_comb)
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

    def train(self, input_data, labels, wandb_logging_callback, verbose, batch_size):
        loss = self.model.fit(input_data, labels, batch_size = batch_size, callbacks = [], verbose=verbose)
        return loss

    def summary(self):
        print(self.model.summary())
