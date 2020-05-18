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
import wandb
from moviepy.editor import *
from moviepy.audio.AudioClip import AudioArrayClip
wandb.init(project='videomelody')

#################################
# Location
#################################
folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\midi\\"
alltime = 0
midi_name = '43pbos12.mid'
midi_path = folder_name + midi_name

#################################
# Read and convert sound array
#################################
pm = pretty_midi.PrettyMIDI(midi_path)
sound_array = pm.synthesize()
ssa = sound_array[0:int(sound_array.shape[0]/50)]

#################################
# Create video array
#################################
video = list()
for a in np.arange(0, 1, 0.01):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:,:,0] = a * 255
    image[:,:,1] = (1-a) * 255
    video.append(image)

#################################
# Create moviepy clips
#################################
ssa = np.reshape(ssa, (-1,1))
clip = ImageSequenceClip(video, fps=25)
aclip = AudioArrayClip(ssa, fps=44100) # from a numerical array
cclip = clip.set_audio(aclip)

# cclip.preview()
# memFile = BytesIO()
cclip.write_videofile("movie.mp4", audio_bitrate = '3000k')
# pm.write(memFile)
# memFile.seek(0)
# playMidiFile(memFile)

#################################
# Encode to base64 an write into html
#################################
f = open('test.html','w')

encoded_data = base64.b64encode(open("movie.mp4", 'rb').read()).decode("utf-8")
video_tag = '<!DOCTYPE html><html><body><video width="400" controls><source type="video/mp4" src="data:video/mp4;base64,{0}"></video></body></html>'.format(encoded_data)

f.write(video_tag)
f.close()

wandb.log({"FINALIE": wandb.Html(open("test.html"))})
