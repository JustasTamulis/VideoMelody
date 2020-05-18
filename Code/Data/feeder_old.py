import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import psutil
import pickle
import random



class feeder:
    def __init__(self, sample_size = 1000, sequence_length = 100, valid_size = 300):

        self.valid_size = valid_size
        self.sample_size = sample_size
        self.sequence_length = sequence_length

        self.folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\"
        hot2piano_path = self.folder_name + "OneHot\\hot2piano.npy"
        piano2hot_path = self.folder_name + "OneHot\\piano2hot.pkl"

        self.hot2piano = np.load(hot2piano_path)
        with open(piano2hot_path, 'rb') as handle:
            self.piano2hot = pickle.load(handle)

        self.order = [2,1,3,7,4,6,5]
        self.ord_point = 0
        # random.shuffle(self.order)
        self.vid_input = None
        self.note_input = None
        self.piano_rolls = None


    def unison_shuffled_copies(self, a, b, c):
        assert len(a) == len(b) == len(c)
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p]


    def get_first_input(self):
        return self.gen_input(True, self.order[0])


    def get_input(self):
        if len(self.vid_input) > self.sample_size + 2*self.sequence_length:
            vid_input, self.vid_input = np.split(self.vid_input, [self.sample_size])
            # note_input, self.note_input = np.split(self.note_input, [self.sample_size])
            piano_rolls, self.piano_rolls = np.split(self.piano_rolls, [self.sample_size])
        else:
            self.ord_point = (self.ord_point + 1) % 8 + 1
            piano_rolls, vid_input, tr, tr, tr, tr, tr = self.gen_input(False, self.ord_point)

        return piano_rolls, vid_input


    def gen_input(self, validation, ord):

        vid_input_all = []
        note_input_all = []
        val_vid_input_all = []
        val_vid_real_all = []
        val_piano_all = []
        piano_rolls_all = []
        gen_roll = []
        needed_samples = self.sample_size
        # samples_by_group = int(needed_samples/7)
        samples_by_group = int(needed_samples)


        ohe_path = self.folder_name + "OneHot\\" + str(ord) + '.npy'
        vec_path = self.folder_name + "img_vec\\" + str(ord) + '.npy'
        vide_path = self.folder_name + "video_npy\\" + str(ord) + '.npy'
        ohe = np.load(ohe_path)
        vec = np.load(vec_path)
        vid = np.load(vide_path)
        vid_input = []
        note_input = []
        val_vid_input = []
        val_vid_real = []
        piano_rolls = []
        val_piano = []
        gen_roll = None
        pianoroll_size = 2502

        current_size = len(ohe)

        if samples_by_group > current_size- 2*self.sequence_length:
            samples_by_group = current_size- 2*self.sequence_length

            # Create one hot encoded vector from number
        onehot = np.zeros((current_size, pianoroll_size))
        for j in range(0, current_size - self.sequence_length):
            note_nr = ohe[j + self.sequence_length]
            note_in = np.zeros(pianoroll_size)
            note_in[note_nr] = 1
            onehot[j] = note_in

            # Create sequences of inputs
        for j in range(self.sequence_length, current_size - self.sequence_length):
            vid_in = vec[j:j + self.sequence_length]
            roll_in = onehot[j:j + self.sequence_length]
            note_in = onehot[j]

            vid_input.append(vid_in)
            # note_input.append(note_in)
            piano_rolls.append(roll_in)

        vid_input = np.reshape(vid_input,(current_size - 2*self.sequence_length,  self.sequence_length, 100))
        # note_input = np.reshape(note_input, (current_size - 2*self.sequence_length, 2502))
        # piano_rolls = np.reshape(piano_rolls, (current_size - self.sequence_length, self.sequence_length, 2502))###
        piano_rolls = np.array(piano_rolls, dtype = np.int8)


        if validation:
            valid_nr = random.randint(10,current_size - 2*self.sequence_length)
            val_vid_input = vid_input[valid_nr:valid_nr+self.valid_size]
            val_vid_real = vid[valid_nr:valid_nr+self.valid_size]
            val_piano = piano_rolls[valid_nr]###
            gen_roll = onehot[valid_nr:valid_nr+self.valid_size]###
            vid_input = np.delete(vid_input, slice(valid_nr,valid_nr+self.valid_size), 0)
            # note_input = np.delete(note_input, slice(valid_nr,valid_nr+self.valid_size), 0)
            piano_rolls = np.delete(piano_rolls, slice(valid_nr,valid_nr+self.valid_size), 0)###


        # vid_input, note_input, piano_rolls = self.unison_shuffled_copies(vid_input, note_input, piano_rolls)
        p = np.random.permutation(len(vid_input))

        vid_input, self.vid_input = np.split(vid_input[p], [samples_by_group])
        # note_input, self.note_input = np.split(note_input[p], [samples_by_group])
        piano_rolls, self.piano_rolls = np.split(piano_rolls[p], [samples_by_group])

        # if i == order[0]:
        vid_input_all = vid_input
        # note_input_all = note_input

        val_vid_input_all = val_vid_input
        val_vid_real_all = val_vid_real

        piano_rolls_all = piano_rolls ###
        val_piano_all = val_piano ###
        # else:
        #     vid_input_all = np.concatenate((vid_input, vid_input_all))
        #     note_input_all = np.concatenate((note_input, note_input_all))
        #     val_vid_input_all = np.concatenate((val_vid_input, val_vid_input_all))
        #     val_vid_real_all = np.concatenate((val_vid_real, val_vid_real_all))
        #
        #     piano_rolls_all = np.concatenate((piano_rolls, piano_rolls_all)) ###

        return piano_rolls_all, vid_input_all, note_input_all, val_piano_all, val_vid_input_all, val_vid_real_all, gen_roll

            # print("DELTED SIZE 1: " + str(val_vid_input.shape))
            # print("DELTED SIZE 2: " + str(val_vid_real.shape))
            # # create input sequences and the corresponding outputs
            # for i in range(self.sample_size - self.sequence_length, self.sample_size - self.sequence_length + self.valid_size):
            #     vid_in = vec[i:i + self.sequence_length]
            #     rl_in = vid[i + self.sequence_length]
            #     val_vid_input.append(vid_in)
            #     val_vid_real.append(rl_in)
            #
            # val_vid_input = np.reshape(val_vid_input,(self.valid_size,  self.sequence_length, 100))
            # val_vid_real = np.reshape(val_vid_real, (self.valid_size, 120,120,3))
