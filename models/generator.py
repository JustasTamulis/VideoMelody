from keras import layers
from keras.models import Model
import numpy as np

# --------------------------------------------------
# ------------------GENERATOR-----------------------
# --------------------------------------------------
class generator():

    def __init__(self, optimizer, random_size, image_shape, roll_size):
        self.optimizer = optimizer
        self.model = self.build_model(random_size, image_shape, roll_size)

    def build_model(self, random_size, image_shape, roll_size):
        # --------------------------------------------------
        # Input random stream
        in_lat = layers.Input(shape=(random_size,), name="Noise_input")
        gen = layers.Dense(28, activation='softmax')(in_lat)
        # --------------------------------------------------
        # Input image
        li_image = layers.Input(shape=image_shape, name="Image_input")
        l_image = layers.Conv2D(27, (4, 4), padding='same', activation='relu')(li_image)
        l_image = layers.MaxPooling2D((4, 4))(l_image)
        l_image = layers.Conv2D(9, (4, 4),  padding='same', activation='relu')(l_image)
        l_image = layers.MaxPooling2D((4, 4))(l_image)
        l_image = layers.Flatten()(l_image)
        l_image = layers.Dropout(0.02)(l_image)
        l_image = layers.Dense(100, activation='relu')(l_image)
        # --------------------------------------------------
        # merge image gen and label input
        merge = layers.Concatenate()([gen, l_image])
        l_comb = layers.Dense(roll_size, activation='relu')(merge)
        out_layer = layers.Dense(roll_size, activation='tanh', name="Note_output")(l_comb) #hard_sigmoid
        # --------------------------------------------------
        # define model
        model = Model([li_image, in_lat], out_layer)
        model.name="Generator"
        return model

    def generate(self, input_data):
         generated_roll = self.model.predict(input_data)
         return generated_roll
