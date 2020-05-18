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
# pianoroll_name = 'roll_ndarray.npy'
# image_name = 'image_ndarray.npy'
result_midi_path = folder_name + 'result_faure.mid'
# midi_path = folder_name + midi_name
avi_path = folder_name + avi_name
piano_path = folder_name + pianoroll_name
image_path = folder_name + image_name
converted_midi_path = folder_name + 'flat_midi.mid'
midi_path = converted_midi_path
# SOUND
# Read MIDI --------------------------------------------------

pm = pretty_midi.PrettyMIDI(midi_path)
print('midi from INPUT RAW end time ' + str(pm.get_end_time()))

multitrack = Multitrack(midi_path, beat_resolution=20)
multitrack.binarize()
fig, axs = multitrack.plot()
plt.show()


# MIDI -> Piano roll --------------------------------------------------

print('piano roll merged length ' + str(multitrack.get_active_length()))
roll = multitrack.get_merged_pianoroll(mode='max')

print('piano roll shape ' +  str(roll.shape))


# MIDI play --------------------------------------------------
# memFile = BytesIO()
# pm.write(memFile)
# memFile.seek(0)
# playMidiFile(memFile)

# AVi play   --------------------------------------------------

clip = VideoFileClip(avi_path)
clip.preview()
# # audio_clip = AudioFileClip(midi_path)
# # audio_clip.preview()
# # clip = clip.set_audio(audio_clip)
#
#
# pygame.display.set_caption('Hello World!')
# pygame.mixer.music.load(midi_path)
# # pygame.mixer.music.play()
# clip.preview()
# # input("<< press anything to stop >>")
# # clip.preview()
#
