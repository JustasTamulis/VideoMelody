import pygame
import os
import matplotlib.pyplot as plt
from moviepy.editor import *
import numpy as np
# from midi2audio import FluidSynth
import FluidSynth
# clip = VideoFileClip('nosound.mp4')
fs = FluidSynth()
pygame.init()

# pygame.mixer.music.load('OneMoreTime.mid')
fs = FluidSynth()
fs.play_midi('OneMoreTime.mid')
# FluidSynth().play_midi('OneMoreTime.mid')
# # mixer.music.load('/dcs/17/u1709355/PROJECT/VideoMelody-master/sound/OneMoreTime.mid')
# pygame.mixer.music.play()
# input("<< press anything to stop >>")
# pygame.mixer.music.stop()

# clip = VideoFileClip('nosound.mp4')
# clip.preview()
