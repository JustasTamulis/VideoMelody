import pygame
import pretty_midi
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from moviepy.editor import *
import cv2
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
midi_name = '1943BossWin.mid'
avi_name = '1943.avi'
midi_path = folder_name + midi_name
avi_path = folder_name + avi_name
# SOUND
# Read MIDI --------------------------------------------------

pyp_mid = Multitrack(midi_path) # beat_resolution=default

pm = pyp_mid.to_pretty_midi()
print('midi from piano roll end time ' + str(pm.get_end_time()))

pyp_mid.remove_empty_tracks()
# print(pyp_mid.tracks)
pyp_mid.merge_tracks(track_indices = [0,1,2,3,4,5,6,7,8,9], mode='max', remove_merged=True)
pyp_mid.binarize()

pm = pretty_midi.PrettyMIDI(midi_path)
print('original pretty midi end time ' + str(pm.get_end_time()))

# MIDI -> Piano roll --------------------------------------------------

print('piano roll merged length ' + str(pyp_mid.get_active_length()))
pyp_pr = pyp_mid.get_merged_pianoroll(mode='max')
# print(pyp_mid.tracks)
print('Piano roll shape ' + str(pyp_pr.shape))
# print(pyp_mid.get_active_pitch_range())
print("........Change piano roll here......")

pm = pyp_mid.to_pretty_midi()
print('midi from piano roll after merge end time ' + str(pm.get_end_time()))
# memFile = BytesIO()
# pm.write(memFile)
# memFile.seek(0)
# playMidiFile(memFile)

# Piano roll -> MIDI --------------------------------------------------

track = Track(pianoroll=pyp_pr, program=0, is_drum=False,
              name='please work')
multitrack = Multitrack(tracks=[track])
# track = multitrack.get_merged_pianoroll(mode='max')
# print(track.shape)

pm = multitrack.to_pretty_midi()
print('midi from piano roll after merge and recover end time ' + str(pm.get_end_time()))
# MIDI to audio_clip --------------------------------------------------


print('############----------VIDEO-------------###############')
# VIDEO
# AVI -> numpay array --------------------------------------------------

vid = vidConvert(avi_path)
print('Video array shape ' + str(vid.shape))

# numpay array -> video_clip --------------------------------------------------



# DISPLAY & OUTPUT
# Play MIDI --------------------------------------------------

# memFile = BytesIO()
# pm.write(memFile)
# memFile.seek(0)
# playMidiFile(memFile)

# Play avi --------------------------------------------------

# audio_clip + video_clip --------------------------------------------------

# Save avi --------------------------------------------------

# writer = cv2.VideoWriter('test1.avi', cv2.VideoWriter_fourcc(*'PIM1'), 25, (640, 480), False)
# for i in range(100):
#     x = np.random.randint(255, size=(480, 640)).astype('uint8')
#     writer.write(x)

# OLD --------------------------------------------------
# pygame.init()
# clip = VideoFileClip(avi_path)
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
# pygame.mixer.music.stop()
# print(pygame.display.get_driver())
#
# # vid = vidConvert(avi_path)
