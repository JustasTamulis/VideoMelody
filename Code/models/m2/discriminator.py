from keras import layers
from keras.models import Model
import numpy as np

# SV_SP

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
        # Input true piano roll
        li_piano = layers.Input(shape=(sequence_length, roll_size), name="Roll_input")
        l_piano = layers.LSTM(64, input_shape=(sequence_length, roll_size),return_sequences=True)(li_piano)
        l_piano = layers.LSTM(32,return_sequences=True)(l_piano)
        l_piano = layers.BatchNormalization()(l_piano)
        # --------------------------------------------------
        # Input condition piano note
        li_cond = layers.Input(shape=(1,roll_size,), name="Pianoroll_input")
        l_cond = layers.Flatten()(l_cond)
        l_cond = layers.Dense(sequence_length*2,activation='relu')(l_cond)
        l_cond = layers.Reshape((sequence_length,2))(l_cond)
        l_cond = layers.BatchNormalization()(l_cond)
        # --------------------------------------------------
        # Input image
        li_image = layers.Input(shape=(sequence_length, image_shape), name="Image_input")
        l_image = layers.LSTM(64, input_shape=(sequence_length, image_shape),return_sequences=True)(li_image)
        l_image = layers.LSTM(32,return_sequences=True)(l_image)
        l_image = layers.BatchNormalization()(l_image)
        # --------------------------------------------------
        # merge
        merge = layers.Concatenate()([l_image, l_piano, l_cond])
        l_comb = layers.LSTM(128,return_sequences=True)(merge)
        l_comb = layers.Bidirectional(layers.LSTM(64))(l_comb)
        l_comb = layers.Dense(100)(l_comb)
        out_layer = layers.Dense(1, activation='sigmoid', name="Discriminator_decision")(l_comb)
        # define model
        model = Model([li_image, li_piano, li_cond], out_layer)
        # compile model
        opt = self.optimizer
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
        model.name="Discriminator"
        return model

    def predict(self, input_data):
         prediction = self.model.predict(input_data)
         return prediction

    def train(self, input_data, labels, wandb_logging_callback, verbose, batch_size):
        loss = self.model.fit(input_data, labels, batch_size = batch_size, callbacks = [], verbose=verbose)
        return loss

    def evaluate(self, input_data, labels, verbose, batch_size):
        loss = self.model.evaluate(input_data, labels, batch_size = batch_size, callbacks = [], verbose=verbose)
        return loss

    def summary(self):
        print(self.model.summary())
