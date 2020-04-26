import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import psutil
import pickle



class feeder:
    def __init__(self, sample_size = 1000, num_video_vec = 100, valid_size = 300):

        self.valid_size = valid_size
        self.sample_size = sample_size
        self.sequence_length = num_video_vec

        self.folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\"
        hot2piano_path = self.folder_name + "OneHot\\hot2piano.npy"
        piano2hot_path = self.folder_name + "OneHot\\piano2hot.pkl"

        self.hot2piano = np.load(hot2piano_path)
        with open(piano2hot_path, 'rb') as handle:
            self.piano2hot = pickle.load(handle)


    def gen_input(self):

        allohe = None
        for i in range(1,2):
            ohe_path = self.folder_name + "OneHot\\" + str(i) + '.npy'
            vec_path = self.folder_name + "img_vec\\" + str(i) + '.npy'
            vide_path = self.folder_name + "video_npy\\" + str(i) + '.npy'
            ohe = np.load(ohe_path)
            vec = np.load(vec_path)
            vid = np.load(vide_path)
            pianoroll_size = 2502
            if self.sample_size > len(ohe):
                self.sample_size = len(ohe)
            print(ohe.shape)
            vid_input = []
            note_input = []
            # create input sequences and the corresponding outputs
            for i in range(0, self.sample_size - self.sequence_length):
                vid_in = vec[i:i + self.sequence_length]
                note_nr = ohe[i + self.sequence_length]
                note_in = np.zeros(pianoroll_size)
                note_in[note_nr] = 1
                vid_input.append(vid_in)
                note_input.append(note_in)

            vid_input = np.reshape(vid_input,(self.sample_size - self.sequence_length,  self.sequence_length, 100))
            note_input = np.reshape(note_input, (self.sample_size - self.sequence_length, 2502))

            val_vid_input = []
            val_vid_real = []
            # create input sequences and the corresponding outputs
            for i in range(self.sample_size - self.sequence_length, self.sample_size - self.sequence_length + self.valid_size):
                vid_in = vec[i:i + self.sequence_length]
                rl_in = vid[i + self.sequence_length]
                val_vid_input.append(vid_in)
                val_vid_real.append(rl_in)

            val_vid_input = np.reshape(val_vid_input,(self.valid_size,  self.sequence_length, 100))
            val_vid_real = np.reshape(val_vid_real, (self.valid_size, 120,120,3))



            return vid_input, note_input, val_vid_input, val_vid_real
