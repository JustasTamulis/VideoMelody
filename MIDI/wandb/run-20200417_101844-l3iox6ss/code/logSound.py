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
from wandb.keras import WandbCallback
from keras.models import load_model
import argparse

wandb.init(project='videomelody')
config = wandb.config

print('---------------------------------')

def playMidiFile(midi_stream):
    pygame.mixer.music.load(midi_stream)
    pygame.mixer.music.play()
    input("<< press anything to stop >>")
    pygame.mixer.music.stop()
    return True


folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\midi\\"
alltime = 0
for i in range(1,2):
    midi_name = str(i)+'.mid'
    midi_path = folder_name + midi_name
    # SOUND
    # Read MIDI --------------------------------------------------

    # pyp_mid = Multitrack(midi_path) # beat_resolution=default
    #
    # pyp_mid.remove_empty_tracks()
    # pyp_mid.merge_tracks(track_indices=[0,1], mode='max', remove_merged=True)
    # pyp_mid.binarize()
    #
    # pm = pyp_mid.to_pretty_midi()
    # print('midi from binarized ' + str(pm.get_end_time()))

    pm = pretty_midi.PrettyMIDI(midi_path)
    print('original pretty midi end time ' + str(pm.get_end_time()))
    alltime = alltime + pm.get_end_time()
    #
    # # Let's look at what's in this MIDI file
    # print('There are {} time signature changes'.format(len(pm.time_signature_changes)))
    # print('There are {} instruments'.format(len(pm.instruments)))
    # # print('Instrument 3 has {} notes'.format(len(pm.instruments[0].notes)))
    # # print('Instrument 4 has {} pitch bends'.format(len(pm.instruments[4].pitch_bends)))
    # # print('Instrument 5 has {} control changes'.format(len(pm.instruments[5].control_changes)))
    # print(pm.instruments)
    #
    #
    # memFile = BytesIO()
    # pm.write(memFile)
    # memFile.seek(0)
    # playMidiFile(memFile)

    sound_array = pm.synthesize()
    wandb.log({"examples": [wandb.Audio(sound_array, caption="Nice", sample_rate=32)]})


print(alltime)
