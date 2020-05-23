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
import pickle

class demonizer:
    def __init__(self, wandb, temp_loc = None):
        self.wandb = wandb
        self.temp_loc = temp_loc
        self.folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\"
        hot2piano_path = self.folder_name + "OneHot\\hot2piano.npy"
        piano2hot_path = self.folder_name + "OneHot\\piano2hot.pkl"

        self.hot2piano = np.load(hot2piano_path)
        with open(piano2hot_path, 'rb') as handle:
            self.piano2hot = pickle.load(handle)


    def log_demo(self, epoch, video, notes):

        #################################
        # convert sound array
        #################################
        generated_roll = np.zeros((len(notes),128))
        # print(notes.shape)


        diff_pitches = set()
        pitch_count = 0

        for i in range(len(notes)):
            nr = np.argmax(notes[i])
            # print(nr)
            diff_pitches.add(nr)
            gen_roll = self.hot2piano[nr]
            generated_roll[i] = gen_roll
        pitch_count = len(diff_pitches)
        track = Track(pianoroll=generated_roll, program=0, is_drum=False,
                      name='Generated midi')
        multitrack = Multitrack(tracks=[track], beat_resolution=25, tempo=120)
        # print("test")
        pm = multitrack.to_pretty_midi()
        # print(pm.get_end_time())
        sound_array = pm.synthesize()
        # sound_array = sound_array[0:-(2*44100)] ############
        # print(generated_roll.shape)
        # print(sound_array.shape)
        fig, axs = multitrack.plot()
        # wandb.log({})
        #################################
        # Create moviepy clips
        #################################
        pygame.init()
        pygame.mixer.quit()

        video = np.reshape(video, (-1, 120,120,3))
        video_list = list()
        for i in range(len(video)):
            video_list.append(video[i])

        # for a in np.arange(0, 1, 0.01):
        #     image = np.zeros((64, 64, 3), dtype=np.uint8)
        #     image[:,:,0] = a * 255
        #     image[:,:,1] = (1-a) * 255
        #     video.append(image)
        # video = video.tolist()
        video = video_list
        # print(type(video))
        # print(type(video[0]))
        # print(video.shape)

        sound_array = np.reshape(sound_array, (-1,1))
        # wave = np.array(ssa, dtype="int64")
        # print("VIDEO SIZE " +  str(len(video)) + " " + str(type(video)))
        clip = ImageSequenceClip(video, fps=25)
        aclip = AudioArrayClip(sound_array, fps=44100) # from a numerical array
        cclip = clip.set_audio(aclip)

        cclip.write_videofile(str(epoch) + "movie.mp4", audio_bitrate = '44100', verbose=False)

        #################################
        # Encode to base64 an write into html
        #################################
        html_name = 'test'+str(epoch)+'.html'
        f = open(html_name,'w')

        encoded_data = base64.b64encode(open(str(epoch) + "movie.mp4", 'rb').read()).decode("utf-8")
        video_tag = '<!DOCTYPE html><html><body>' + str(epoch) + '<video width="400" controls><source type="video/mp4" src="data:video/mp4;base64,{0}"></video></body></html>'.format(encoded_data)

        f.write(video_tag)
        f.close()

        # self.wandb.log({"Epoch " +str(epoch): wandb.Html(open(html_name)), "roll " + str(epoch): fig})
        try:
          self.wandb.log({"Epoch " +str(epoch): wandb.Html(open(html_name)), "roll " + str(epoch): fig, "validation pitch count": pitch_count})
        except:
          print("An exception occurred")
