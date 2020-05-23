import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import psutil
import pickle
import random



class feeder:
    def __init__(self, model_nr, sample_size = 1000, valid_size = 1):

        if sample_size > 14000:
            sample_size = 140000

        if valid_size > 140:
            valid_size = 140

        self.model_nr = model_nr
        self.valid_size = valid_size
        self.sample_size = sample_size

        # folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\"
        folder_name = "C:\\VideoMelody\\"
        self.train_folder = folder_name + "train\\"
        self.test_folder = folder_name + "test\\"


    def get_validation(self):
        valid_nr = np.random.randint(0, 140, self.valid_size)
        SV = []
        P = []
        SP = []
        SPt = []
        SRV = []

        if self.valid_size > 80:
            SV = np.load(self.test_folder + "SV\\all" + '.npy')
            P = np.load(self.test_folder + "P\\all" + '.npy')
            SP = np.load(self.test_folder + "SP\\all" + '.npy')
            SPt = np.load(self.test_folder + "SPt\\all" + '.npy')
            SRV = np.load(self.test_folder + "SRV\\all" + '.npy')
        else:
            SV = np.zeros((self.valid_size, 50, 100), dtype=np.float32)
            P = np.zeros((self.valid_size, 1, 2502), dtype=np.uint8)
            SP = np.zeros((self.valid_size, 50, 2502), dtype=np.uint8)
            SPt = np.zeros((self.valid_size, 50, 2502), dtype=np.uint8)
            SRV = np.zeros((self.valid_size, 50, 120, 120, 3), dtype=np.float32)

            for j in range(self.valid_size):
                i = valid_nr[j]
                sv = np.load(self.test_folder + "SV\\" + str(i) + '.npy')
                p = np.load(self.test_folder + "P\\" + str(i) + '.npy')
                sp = np.load(self.test_folder + "SP\\" + str(i) + '.npy')
                spt = np.load(self.test_folder + "SPt\\" + str(i) + '.npy')
                srv = np.load(self.test_folder + "SRV\\" + str(i) + '.npy')
                SV[j] = sv
                P[j] = p
                SP[j] = sp
                SPt[j] = spt
                SRV[j] = srv

            SV = np.array(SV, dtype = np.float32)
            P = np.array(P, dtype = np.int8)
            SP = np.array(SP, dtype = np.int8)
            SPt = np.array(SPt, dtype = np.int8)
            SRV = np.array(SRV, dtype = np.float32)

        # np.save(self.test_folder + "SV\\all" + '.npy', SV)
        # np.save(self.test_folder + "P\\all" + '.npy', P)
        # np.save(self.test_folder + "SP\\all" + '.npy', SP)
        # np.save(self.test_folder + "SPt\\all" + '.npy', SPt)
        # np.save(self.test_folder + "SRV\\all" + '.npy', SRV)

        return SV, P, SP, SPt, SRV


    def get_input(self):


        if self.sample_size > 10000:
            SV = np.load(self.train_folder + "SV\\all" + '.npy')
            P = np.load(self.train_folder + "P\\all" + '.npy')
            SP = np.load(self.train_folder + "SP\\all" + '.npy')
            SPt = np.load(self.train_folder + "SPt\\all" + '.npy')
        else:
            SV = np.zeros((self.sample_size, 50, 100), dtype=np.float32)
            P = np.zeros((self.sample_size, 1, 2502), dtype=np.uint8)
            SP = np.zeros((self.sample_size, 50, 2502), dtype=np.uint8)
            SPt = np.zeros((self.sample_size, 50, 2502), dtype=np.uint8)
            valid_nr = np.random.randint(0, 14000, self.sample_size)
            for j in range(self.sample_size):
                i = valid_nr[j]
                sv = np.load(self.train_folder + "SV\\" + str(i) + '.npy')
                p = np.load(self.train_folder + "P\\" + str(i) + '.npy')
                sp = np.load(self.train_folder + "SP\\" + str(i) + '.npy')
                spt = np.load(self.train_folder + "SPt\\" + str(i) + '.npy')
                SV[j] = sv
                P[j] = p
                SP[j] = sp
                SPt[j] = spt

            SV = np.array(SV, dtype = np.float32)
            P = np.array(P, dtype = np.int8)
            SP = np.array(SP, dtype = np.int8)
            SPt = np.array(SPt, dtype = np.int8)

        # np.save(self.train_folder + "SV\\all" + '.npy', SV)
        # np.save(self.train_folder + "P\\all" + '.npy', P)
        # np.save(self.train_folder + "SP\\all" + '.npy', SP)
        # np.save(self.train_folder + "SPt\\all" + '.npy', SPt)

        return SV, P, SP, SPt
