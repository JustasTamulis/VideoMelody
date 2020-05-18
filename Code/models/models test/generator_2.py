from keras import layers
from keras.models import Model
import numpy as np
from keras.layers import CuDNNLSTM


# --------------------------------------------------
# (sequence_length, random_size) + (sequence_length, video_vec_size) -> (sequence_length, roll_size)
# --------------------------------------------------
# ------------------GENERATOR-----------------------
# --------------------------------------------------
class generator():

    def __init__(self, optimizer, random_size, image_shape, roll_size, sequence_length):
        self.optimizer = optimizer
        self.model = self.build_model(random_size, image_shape, roll_size, sequence_length)

    def build_model(self, random_size, image_shape, roll_size, sequence_length):
        lstm_size = 32
        # Input piano roll :o
        li_piano = layers.Input(shape=(sequence_length, roll_size), name="Roll_input")
        l_piano = layers.LSTM(64, input_shape=(sequence_length, roll_size),return_sequences=True)(li_piano)
        l_piano = layers.Bidirectional(layers.LSTM(lstm_size,return_sequences=True))(l_piano)
        # l_piano = layers.BatchNormalization()(l_piano)
        # l_piano = layers.Dense(100, activation='tanh')(l_piano)
        # l_piano = layers.LeakyReLU(alpha=0.1)(l_piano)
        # l_piano = layers.Dropout(0.05)(l_piano)
        # l_piano = layers.Dense(50, activation='tanh')(l_piano)
        # --------------------------------------------------
        # Input random stream
        li_ran = layers.Input(shape=(random_size,), name="Noise_input")
        # l_rand = layers.UpSampling1D(size=4)(li_ran)
        # l_ran = layers.Conv1D(filters = 4, kernel_size = int(random_size/4)) (li_ran)
        l_ran = layers.Dense(sequence_length*2)(li_ran)
        l_ran = layers.Reshape((sequence_length, 2))(l_ran)
        # --------------------------------------------------
        # Input image
        li_image = layers.Input(shape=(sequence_length, image_shape), name="Image_input")
        l_image = layers.LSTM(64, input_shape=(sequence_length, image_shape),return_sequences=True)(li_image)
        l_image = layers.Bidirectional(layers.LSTM(lstm_size, return_sequences=True))(l_image)
        # l_image = layers.BatchNormalization()(l_image)
        # l_image = layers.Dense(100, activation='relu')(l_image)
        # l_image = layers.LeakyReLU(alpha=0.1)(l_image)
        # l_image = layers.Dropout(0.05)(l_image)
        # l_image = layers.Dense(100, activation='relu')(l_image)
        # l_image = layers.Flatten()(l_image)
        # --------------------------------------------------
        # merge image gen and label input
        merge = layers.Concatenate()([l_ran, l_image, l_piano])
        # l_comb = layers.BatchNormalization()(merge)
        # l_comb = layers.Dropout(0.05)(l_comb)
        # l_comb = layers.Dense(300, activation='relu')(l_comb)
        # l_comb = layers.Flatten()(l_comb)
        # l_comb = layers.reshape((100,4))(l_comb)
        # l_comb = layers.Convolutional2d((100,4))(l_comb) (100,4) -> (100,2502)
        l_comb = layers.LSTM(64,return_sequences=True)(merge)
        l_comb = layers.Bidirectional(layers.LSTM(128))(l_comb)
        # l_comb = layers.Dropout(0.05)(l_comb)
        # soft_out = Activation('softmax')(input_tensor)
        out_layer = layers.Dense(roll_size, activation='softmax', name="Note_output")(l_comb)
        # --------------------------------------------------
        # define model
        model = Model([li_image, li_ran, li_piano], out_layer)
        model.name="Generator"
        return model

    def generate(self, input_data):
         generated_roll = self.model.predict(input_data)
         return generated_roll

    def get_model(self):
         return self.model
