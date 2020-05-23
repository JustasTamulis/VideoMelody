import pygame
import pretty_midi
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from moviepy.editor import *
import numpy as np
from pypianoroll import Multitrack, Track
from io import BytesIO
print('---------------------------------')

def playMidiFile(midi_stream):
    pygame.mixer.music.load(midi_stream)
    pygame.mixer.music.play()
    input("<< press anything to stop >>")
    pygame.mixer.music.stop()
    return True


folder_name = "F:\\CompSci\\project\\MIDI\\Piano midi\\"
checkpoint_dir = folder_name + "checkpoints\\"
midi_name = 'faure-dolly-suite-1-berceuse.mid'
avi_name = 'faure.avi'
pianoroll_name = 'roll_ndarray.npy'
image_name = 'image_ndarray.npy'
result_midi_path = folder_name + 'result_faure.mid'
midi_path = folder_name + midi_name
avi_path = folder_name + avi_name
piano_path = folder_name + pianoroll_name
image_path = folder_name + image_name
converted_midi_path = folder_name + 'flat_midi.mid'
# SOUND
# Read MIDI --------------------------------------------------

pm = pretty_midi.PrettyMIDI(midi_path)
print('midi from INPUT RAW end time ' + str(pm.get_end_time()))

if pm.time_signature_changes:
    pm.time_signature_changes.sort(key=lambda x: x.time)
    first_beat_time = pm.time_signature_changes[0].time
else:
    first_beat_time = pm.estimate_beat_start()


beat_times = pm.get_beats(first_beat_time)
n_beats = len(beat_times)
print('n_beats ' + str(n_beats))

# print(pm.estimate_tempi())
pyp_mid = Multitrack(midi_path, beat_resolution=20)
pyp_mid.binarize()
print(str(pyp_mid))
average_tempo = np.mean(pyp_mid.tempo)
# pyp_mid.remove_empty_tracks()
# pyp_mid.merge_tracks(track_indices = [0,1,2,3,4,5,6,7,8,9], mode='max', remove_merged=True)
# pyp_mid.binarize()

pm = pyp_mid.to_pretty_midi()
print('midi from INPUT multitrack end time ' + str(pm.get_end_time()))

# MIDI -> Piano roll --------------------------------------------------

print('piano roll merged length ' + str(pyp_mid.get_active_length()))
pyp_pr = pyp_mid.get_merged_pianoroll(mode='max')

# INPPUT TO NETWORK
print("Neural:")

print(pyp_pr.shape)

print("______________________")
# OUTPUT FROM NETWORK
out_pr = pyp_pr


# Piano roll -> MIDI --------------------------------------------------

track = Track(pianoroll=out_pr, program=0, is_drum=False,
              name='please work')
multitrack = Multitrack(tracks=[track], beat_resolution=20, tempo = average_tempo)
# track = multitrack.get_merged_pianoroll(mode='max')
# print(track.shape)
print(average_tempo)
pm = multitrack.to_pretty_midi()
print('midi from OUTPUT piano roll end time ' + str(pm.get_end_time()))
if pm.time_signature_changes:
    pm.time_signature_changes.sort(key=lambda x: x.time)
    first_beat_time = pm.time_signature_changes[0].time
else:
    first_beat_time = pm.estimate_beat_start()

# np.save(piano_path, out_pr)
print(out_pr.shape)
beat_times = pm.get_beats(first_beat_time)
n_beats = len(beat_times)
print('n_beats ' + str(n_beats))

# MIDI play --------------------------------------------------
# memFile = BytesIO()
# pm.write(memFile)
# memFile.seek(0)
# playMidiFile(memFile)

# pm.write(converted_midi_path)

while os.path.isfile(converted_midi_path) == False:
    time.sleep(1)
print('Converted MIDI saved')

pm = pretty_midi.PrettyMIDI(converted_midi_path)
print('midi from CONVERT INPUT end time ' + str(pm.get_end_time()))
