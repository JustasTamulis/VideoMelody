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


folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\test\\"
midi_name = '1.mid'
pianoroll_name = '1.npy'
midi_path = folder_name + midi_name
piano_path = folder_name + pianoroll_name
# SOUND
# Read MIDI --------------------------------------------------

print('Original midi \n')
pm = pretty_midi.PrettyMIDI(midi_path)
print('pretty midi from INPUT RAW end time ' + str(pm.get_end_time()))
# pyp_mid = Multitrack()
# pyp_mid.parse_pretty_midi(pm)

print('\n Piano roll \n')
pyp_mid = Multitrack(midi_path, beat_resolution=25)
print(str(pyp_mid.tempo.shape) + '  - tempo shape')


print('\n Binary roll \n')
pyp_mid.remove_empty_tracks()
track_indices = [idx for idx, track in enumerate(pyp_mid.tracks)]
print(track_indices)
pyp_mid.merge_tracks(track_indices = track_indices, mode='max', remove_merged=True)
pyp_mid.binarize()
print('Active length ' + str(pyp_mid.get_active_length()))
print('BEAT RESOLUTION ' + str(pyp_mid.beat_resolution))

print('\n Midi from roll \n')
pm = pyp_mid.to_pretty_midi()
print('pretty midi from binary piano roll end time ' + str(pm.get_end_time()))

if pm.time_signature_changes:
    pm.time_signature_changes.sort(key=lambda x: x.time)
    first_beat_time = pm.time_signature_changes[0].time
else:
    first_beat_time = pm.estimate_beat_start()

beat_times = pm.get_beats(first_beat_time)
n_beats = len(beat_times)
print('n_beats ' + str(n_beats))

# MIDI -> Piano roll --------------------------------------------------

print('piano roll merged length ' + str(pyp_mid.get_active_length()))
pyp_pr = pyp_mid.get_merged_pianoroll(mode='max')

np.save(piano_path, pyp_pr)
pload = np.load(piano_path)

# Piano roll -> MIDI --------------------------------------------------

track = Track(pianoroll=pload, program=0, is_drum=False,name='please work')
multitrack = Multitrack(tracks=[track])
track = multitrack.get_merged_pianoroll(mode='max')
print(track.shape)
print('Active length ' + str(multitrack.get_active_length()))
print(str(multitrack.tempo) + '  - tempo')
pm = multitrack.to_pretty_midi()
print('midi from OUTPUT  end time ' + str(pm.get_end_time()))

if pm.time_signature_changes:
    pm.time_signature_changes.sort(key=lambda x: x.time)
    first_beat_time = pm.time_signature_changes[0].time
else:
    first_beat_time = pm.estimate_beat_start()


beat_times = pm.get_beats(first_beat_time)
n_beats = len(beat_times)
print('n_beats ' + str(n_beats))

# MIDI play --------------------------------------------------
memFile = BytesIO()
pm.write(memFile)
memFile.seek(0)
playMidiFile(memFile)

# MIDI save --------------------------------------------------
# pm.write(converted_midi_path)
#
# while os.path.isfile(converted_midi_path) == False:
#     time.sleep(1)
# print('Converted MIDI saved')
#
# pm = pretty_midi.PrettyMIDI(converted_midi_path)
# print('midi from CONVERT INPUT end time ' + str(pm.get_end_time()))
