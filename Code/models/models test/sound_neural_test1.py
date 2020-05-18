# Midi load and save piano roll


import pygame
import pretty_midi
import tensorflow as tf
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from moviepy.editor import *
import numpy as np
from pypianoroll import Multitrack, Track
from io import BytesIO
import cv2
print('---------------------------------')

def playMidiFile(midi_stream):
    pygame.mixer.music.load(midi_stream)
    pygame.mixer.music.play()
    input("<< press anything to stop >>")
    pygame.mixer.music.stop()
    return True

def vidConvert(v_path):
    cap = cv2.VideoCapture(v_path)

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.

    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()
    return buf


folder_name = "F:\\CompSci\\project\\MIDI\\ungrouped midi\\"
midi_name = 'convert_1943BossWin.mid'
avi_name = '1943.avi'
pianoroll_name = '1943.npy'
midi_path = folder_name + midi_name
avi_path = folder_name + avi_name
piano_path = folder_name + pianoroll_name
# SOUND
# Read MIDI --------------------------------------------------

pm = pretty_midi.PrettyMIDI(midi_path)
print('midi from INPUT RAW end time ' + str(pm.get_end_time()))
# print(pm.estimate_tempi())

# memFile = BytesIO()
# pm.write(memFile)
# memFile.seek(0)
# playMidiFile(memFile)

pyp_mid = Multitrack()
pyp_mid.parse_pretty_midi(pm)

# if pm.time_signature_changes:
#     pm.time_signature_changes.sort(key=lambda x: x.time)
#     first_beat_time = pm.time_signature_changes[0].time
# else:
#     first_beat_time = pm.estimate_beat_start()
#
#
# beat_times = pm.get_beats(first_beat_time)
# n_beats = len(beat_times)
# print('n_beats ' + str(n_beats))


# pyp_mid = Multitrack(midi_path, beat_resolution=25)
# print(str(pyp_mid.tempo.shape) + '  - tempo')
# pyp_mid.remove_empty_tracks()
# pyp_mid.merge_tracks(track_indices = [0,1,2,3,4,5,6,7,8,9], mode='max', remove_merged=True)
# pyp_mid.binarize()
print('Active length ' + str(pyp_mid.get_active_length()))
pm = pyp_mid.to_pretty_midi()
print('midi from INPUT multitrack end time ' + str(pm.get_end_time()))
print('BEAR RESOLUTION ' + str(pyp_mid.beat_resolution))

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

# notes = pyp_pr
# maxnote = np.max(notes)
# for note in notes[:1000]:
#     cnote = []
#     for i in range(note.shape[0]):
#         if(note[i] > 0):
#             cnote.append(note[i])
#     # if cnote != []:
#     print(cnote)
# print(maxnote)

np.save(piano_path, pyp_pr)
print(type(pyp_pr))
pload = np.load(piano_path)

# INPPUT TO NETWORK
print("______________________")
print("Neural:")
print(pyp_pr.shape)
notes = pyp_pr
sequence_length = 100
network_input = []
network_output = []
# create input sequences and the corresponding outputs
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append(sequence_in)
    network_output.append(sequence_out)
n_patterns = len(network_input)
# reshape the input into a format compatible with LSTM layers
print(n_patterns)

network_input = np.reshape(network_input, (n_patterns, sequence_length, 128))
# normalize input
# network_input = network_input / float(n_vocab)
# network_output = np_utils.to_categorical(network_output)
print(network_input.shape)
# print(network_input)

# define the LSTM model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2])))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

generator = make_generator_model()
# print(generator.summary())


pattern = network_input[0]
print(len(pattern))
x = np.reshape(pattern, (1, 100, 128)).astype('float32')

prediction = generator(x, training=False)


# print(prediction)
print("______________________")
# OUTPUT FROM NETWORK
out_pr = pyp_pr

# Piano roll -> MIDI --------------------------------------------------

track = Track(pianoroll=out_pr, program=0, is_drum=False,
              name='please work')
multitrack = Multitrack(tracks=[track])
# track = multitrack.get_merged_pianoroll(mode='max')
# print(track.shape)
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

print('--------------------------------------------------')
#
# vid = vidConvert(avi_path)
# print('Video array shape ' + str(vid.shape))
# clip = VideoFileClip(avi_path)
# print( clip.duration )
