import sys
import os
import subprocess
from Autoencoder.cae import ConvAutoEncoder
from models.m1.GAN import GAN1 as GAN
import numpy as np
import pickle
import cv2
from moviepy.editor import *
from moviepy.audio.AudioClip import AudioArrayClip
from pypianoroll import Multitrack, Track
import pretty_midi

#  Code to read video file to numpy
def getVideo(name):
    cap = cv2.VideoCapture(name)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frameCount-100  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    return buf

# Function to convert piano roll into pretty_midi object.
def roll2midi(roll):
    generated_roll = np.zeros((len(roll),128))
    for i in range(len(roll)):
        nr = np.argmax(roll[i])
        gen_roll = hot2piano[nr]
        generated_roll[i] = gen_roll
    track = Track(pianoroll=generated_roll, program=0, is_drum=False)
    multitrack = Multitrack(tracks=[track], beat_resolution=25, tempo=60)
    pm = multitrack.to_pretty_midi()
    for notes in [instrument.notes for instrument in pm.instruments]:
        for note in notes:
            note.velocity = 100
    return pm

#################################
# Resize given video
#################################
inFile = sys.argv[1]
subprocess.call(['ffmpeg', '-i', inFile, '-vf', 'scale=120:120', 'temp.avi'])

#################################
# Load the encoder
#################################
cae = ConvAutoEncoder(input_shape=(120,120,3), output_dim=100)
cae.load_weights(path = 'weights', prefix = "koya_")

#################################
# Load and prepare video as normalized numpy array and encode it
#################################
video = getVideo('temp.avi')
video = video.astype(np.float32)
video = video / 255
vec = cae.encode(video)

#################################
# Initialize GAN
#################################
pianoroll_size = 2502
video_vec_size = 100
sequence_length = 50
random_size = sequence_length * 128
gan = GAN(random_size, pianoroll_size, video_vec_size, sequence_length)
sample_weights_gen = "weights\\generator.h5"
sample_weights_dis = "weights\\discriminator.h5"
gan.load_weights(sample_weights_gen, sample_weights_dis)

#################################
# Generate one hot encoed piano roll
#################################
gen_roll = np.zeros((len(vec), pianoroll_size))
for i in range(0, len(vec) - sequence_length, sequence_length):
    noise = np.random.normal(size=(1, random_size))
    vid_in = np.reshape(vec[i:i+sequence_length], (1, sequence_length, video_vec_size))
    g_roll = gan.generate([vid_in, noise])
    g_roll = np.reshape(g_roll, (sequence_length, pianoroll_size))
    gen_roll[i:i+sequence_length] = g_roll

#################################
# Convert it to MIDI, then to raw audio form
#################################
hot2piano_path = "weights\\hot2piano.npy"
hot2piano = np.load(hot2piano_path)
midi = roll2midi(gen_roll)
sound_array = midi.synthesize()


#################################
# Create moviepy clips
#################################
video = getVideo(inFile)
print(video.shape)
video_list = list()
for i in range(len(video)):
    video_list.append(video[i])
video = video_list

sound_array = np.reshape(sound_array, (-1,1))
clip = ImageSequenceClip(video, fps=25)
aclip = AudioArrayClip(sound_array, fps=44100) # from a numerical array
cclip = clip.set_audio(aclip)
cclip.write_videofile("Enchanced.mp4", audio_bitrate = '44100', verbose=False)
