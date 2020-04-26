# Make binary piano rolls from midi, then save it as midi, npy and mp3

# Midi load and save piano roll

import pygame
import pretty_midi
import os
import matplotlib.pyplot as plt
from moviepy.editor import *
import numpy as np
from pypianoroll import Multitrack, Track
from io import BytesIO
import cv2

def playMidiFile(midi_stream):
    pygame.mixer.music.load(midi_stream)
    pygame.mixer.music.play()
    input("<< press anything to stop >>")
    pygame.mixer.music.stop()
    return True


for i in range(1,8):
    folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\"
    midi_name =  "midi\\" + str(i) + '.mid'
    pianoroll_name = "midi_npy\\" + str(i) + '.npy'
    midi_path = folder_name + midi_name
    piano_path = folder_name + pianoroll_name
    # SOUND
    # Read MIDI --------------------------------------------------

    print('Original midi \n')
    pm = pretty_midi.PrettyMIDI(midi_path)
    print('pretty midi from INPUT RAW end time ' + str(pm.get_end_time()))



    pyp_mid = Multitrack()
    pyp_mid.parse_pretty_midi(pm, binarized = True)
    # parse_mid = parse(midi_path,  beat_resolution=25)
    pyp_mid.pad_to_same()
    av_temp = np.mean(pyp_mid.tempo)
    print("average tempo " + str(av_temp))

    pr = pyp_mid.get_merged_pianoroll(mode='max')
    print(pr.shape)


    np.save(piano_path, pr)
    pload = np.load(piano_path)

    # Piano roll -> MIDI --------------------------------------------------

    track = Track(pianoroll=pload, program=0, is_drum=False,name='please work')
    multitrack = Multitrack(tracks=[track])
    track = multitrack.get_merged_pianoroll(mode='max')
    print(track.shape)

    pm = multitrack.to_pretty_midi(constant_tempo = av_temp)
    print('midi from OUTPUT  end time ' + str(pm.get_end_time()))

# MIDI play --------------------------------------------------
# memFile = BytesIO()
# pm.write(memFile)
# memFile.seek(0)
# playMidiFile(memFile)

# MIDI save --------------------------------------------------
# pm.write(converted_midi_path)
#
# while os.path.isfile(converted_midi_path) == False:
#     time.sleep(1)
# print('Converted MIDI saved')
#
# pm = pretty_midi.PrettyMIDI(converted_midi_path)
# print('midi from CONVERT INPUT end time ' + str(pm.get_end_time()))
