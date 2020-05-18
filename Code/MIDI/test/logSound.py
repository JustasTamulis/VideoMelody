import pygame
import pretty_midi
import os
import matplotlib.pyplot as plt
from moviepy.editor import *
import cv2
import numpy as np
from pypianoroll import Multitrack, Track
from io import BytesIO

import wandb
import argparse


wandb.init(project='videomelody')

folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\midi\\"
alltime = 0
midi_name = '43pbos12.mid'
midi_path = folder_name + midi_name

pm = pretty_midi.PrettyMIDI(midi_path)
sound_array = pm.synthesize()
ssa = sound_array[0:int(sound_array.shape[0]/10)]

wandb.log({"examples": [wandb.Audio(ssa, caption=" sample_rate=44100", sample_rate=44100)]})

video = list()

for a in np.arange(0, 1, 0.01):
    image = np.zeros((3, 64, 64), dtype=np.uint8)
    image[0] = a * 255
    image[2] = (1-a) * 255

    video.append(image)
