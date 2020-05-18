import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import psutil
import pickle
import random

valid_size = 20
sample_size = 2000
sequence_length = 50
pianoroll_size = 2502

folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\"
order = [1,2,3,4,5,6,7]
# order = [2]

for ord in order:
    ohe_path = folder_name + "OneHot\\" + str(ord) + '.npy'
    vec_path = folder_name + "img_vec\\" + str(ord) + '.npy'
    vide_path = folder_name + "video_npy\\" + str(ord) + '.npy'

    ohe = np.load(ohe_path)
    vec = np.load(vec_path)
    vid = np.load(vide_path)

    SV = []
    P = []
    SP = []
    SPt = []

    SV_val = []
    P_val = []
    SP_val = []
    SPt_val = []
    SRV = []

    current_size = len(ohe)

        # Create one hot encoded for notes
    onehot = np.zeros((current_size, pianoroll_size))
    for j in range(0, current_size - sequence_length):
        note_nr = ohe[j + sequence_length]
        note_in = np.zeros(pianoroll_size)
        note_in[note_nr] = 1
        onehot[j] = note_in

    valid_nr = np.random.randint(sequence_length, current_size - sequence_length, valid_size*2)

        # Create sequences of inputs
    anomolies = 0
    for j in range(sequence_length, current_size - sequence_length + anomolies):

        sv = vec[j:j + sequence_length]
        p = onehot[j]
        sp = onehot[j:j + sequence_length]
        spt = onehot[j-sequence_length:j]
        srv = vid[j:j+sequence_length]

        if 0 in srv.shape:
            anomolies = anomolies + 1
            # if j in valid_nr:
            #     valid_nr[np.where(valid_nr==j)[0][0]] =
        else:
            if j in valid_nr:
                SV_val.append(sv)
                P_val.append(p)
                SP_val.append(sp)
                SPt_val.append(spt)
                SRV.append(srv)

            else:
                SV.append(sv)
                P.append(p)
                SP.append(sp)
                SPt.append(spt)
    print(anomolies)
    # vid_input = np.reshape(vid_input,(current_size - 2*self.sequence_length,  self.sequence_length, 100))
    # note_input = np.reshape(note_input, (current_size - 2*self.sequence_length, 2502))
    # piano_rolls = np.reshape(piano_rolls, (current_size - self.sequence_length, self.sequence_length, 2502))
    SV = np.array(SV, dtype = np.float32)
    P = np.array(P, dtype = np.int8)
    SP = np.array(SP, dtype = np.int8)
    SPt = np.array(SPt, dtype = np.int8)

    p = np.random.permutation(len(SV))

    SV = SV[p][0:sample_size]
    P = P[p][0:sample_size]
    SP = SP[p][0:sample_size]
    SPt = SPt[p][0:sample_size]

    for j in range(sample_size):
        np.save(folder_name + "train\\SV\\" + str(j+(ord-1)*sample_size) + '.npy', SV[j])
        np.save(folder_name + "train\\P\\" + str(j+(ord-1)*sample_size) + '.npy', P[j])
        np.save(folder_name + "train\\SP\\" + str(j+(ord-1)*sample_size) + '.npy', SP[j])
        np.save(folder_name + "train\\SPt\\" + str(j+(ord-1)*sample_size) + '.npy', SPt[j])

    for j in range(valid_size):
        np.save(folder_name + "test\\SV\\" + str(j+(ord-1)*valid_size) + '.npy', SV_val[j])
        np.save(folder_name + "test\\P\\" + str(j+(ord-1)*valid_size) + '.npy', P_val[j])
        np.save(folder_name + "test\\SP\\" + str(j+(ord-1)*valid_size) + '.npy', SP_val[j])
        np.save(folder_name + "test\\SPt\\" + str(j+(ord-1)*valid_size) + '.npy', SPt_val[j])
        np.save(folder_name + "test\\SRV\\" + str(j+(ord-1)*valid_size) + '.npy', SRV[j])
