from scipy.io.wavfile import write
import base64
import pygame
import pretty_midi
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pypianoroll import Multitrack, Track
import argparse
from io import BytesIO
from moviepy.editor import *
from moviepy.audio.AudioClip import AudioArrayClip


#################################
# Location
#################################
folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\midi\\"
alltime = 0
midi_name = '1.mid'
midi_path = folder_name + midi_name

#################################
# Read and convert sound array
#################################
pm = pretty_midi.PrettyMIDI(midi_path)
sound_array = pm.synthesize()
samplerate = 44100
#################################
# Create moviepy clips
#################################
pygame.init()
pygame.mixer.quit()
ssa = np.reshape(sound_array, (-1,1))
aclip = AudioArrayClip(ssa, fps=44100) # from a numerical array

aclip.write_audiofile("test.mp3")
