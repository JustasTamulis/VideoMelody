import pygame
import pretty_midi
import tensorflow as tf
import os
from os import startfile
import matplotlib.pyplot as plt
from moviepy.editor import *
import numpy as np
# import librosa.display
from pypianoroll import Multitrack, Track

# from StringIO import StringIO
from io import BytesIO ## for Python 3

folder_name = "F:\\CompSci\\project\\MIDI\\DaftPunk\\"
# midi_name = 'OneMoreTime'
midi_name = 'HarderBetterFasterStronger'
# midi_name = 'Voyager'
midi_path = folder_name + midi_name + '//' + midi_name + '.mid'

test_mid1 = folder_name + midi_name + '//' + "this_is_a_test" + '.mid'
test_mid2 = folder_name + midi_name + '//' + "test" + '.mid'
test_mid3 = folder_name + midi_name + '//' + "ofoct" + '.mid'
# pygame.init()

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

def get_piano_roll_mod(midi, fs=100, times=None, pedal_threshold=64):
    """Compute a piano roll matrix of the MIDI data.
    Parameters
    ----------
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    times : np.ndarray
        Times of the start of each column in the piano roll.
        Default ``None`` which is ``np.arange(0, get_end_time(), 1./fs)``.
    pedal_threshold : int
        Value of control change 64 (sustain pedal) message that is less
        than this value is reflected as pedal-off.  Pedals will be
        reflected as elongation of notes in the piano roll.
        If None, then CC64 message is ignored.
        Default is 64.
    Returns
    -------
    piano_roll : np.ndarray, shape=(128,times.shape[0])
        Piano roll of MIDI data, flattened across instruments.
    """

    # If there are no instruments, return an empty array
    if len(midi.instruments) == 0:
        return np.zeros((128, 0))

    # Remove effects and odd instruments.
    ii = 0
    for inst in midi.instruments:
        if inst.program > 96 or inst.program == 0:
             midi.instruments.pop(ii)
        else:
            ii += 1

    # Get piano rolls for each instrument
    piano_rolls = [i.get_piano_roll(fs=fs, times=times)
                   for i in midi.instruments]

    # Allocate piano roll,
    # number of columns is max of # of columns in all piano rolls
    piano_roll = np.zeros((128, np.max([p.shape[1] for p in piano_rolls])))
    # Sum each piano roll into the aggregate piano roll
    for roll in piano_rolls:
        piano_roll[:, :roll.shape[1]] += roll

    piano_roll[piano_roll > 0] = 60

    return piano_roll

def playMidiFile(midi_stream):
    pygame.mixer.music.load(midi_stream)
    pygame.mixer.music.play()
    input("<< press anything to stop >>")
    pygame.mixer.music.stop()
    return True


class Video(object):
    def __init__(self, folder, name):
        self.folder = folder + name + "\\"
        self.name = name
        self.path = self.folder + name + '.mp4'
        self.sound_path = self.folder + name + '.mid'
        self.cutpath = self.folder + name + '_cut.mp4'
        self.clip = VideoFileClip(self.path)
        pm = pretty_midi.PrettyMIDI(self.sound_path)
        pygame.mixer.music.load(self.sound_path)

    def play(self):
        pygame.display.set_caption(self.name)
        pygame.mixer.music.play()
        self.clip.preview()

    def play(self):
        pygame.display.set_caption(self.name)
        pygame.mixer.music.play()
        self.clip.preview()

    def play_cut(self):
        pygame.display.set_caption(self.name + ' CUT')
        pygame.mixer.music.play()
        self.clip_cut.preview()

    def cut(self, start_time, end_time):
        self.clip_cut = self.clip.subclip(start_time,end_time)


playMidiFile(test_mid3)

# movie = Video(folder_name, midi_name)
#
# pyp_mid = Multitrack(midi_path)
# pyp_mid.remove_empty_tracks()
# # print(pyp_mid.tracks)
# pyp_mid.merge_tracks(track_indices = [0,1,2,3,4,5,6,7,8,9], mode='max', remove_merged=True)
# pyp_mid.binarize()
# pyp_pr = pyp_mid.get_merged_pianoroll(mode='max')
# # print(pyp_mid.tracks)
# print(pyp_pr.shape)
# # print(pyp_mid.get_active_pitch_range())
# print("..............")
#
# # pm = pyp_mid.to_pretty_midi()
# # memFile = BytesIO()
# # pm.write(memFile)
# # memFile.seek(0)
# # playMidiFile(memFile)
#
#
# track = Track(pianoroll=pyp_pr, program=0, is_drum=False,
#               name='please work')
# multitrack = Multitrack(tracks=[track])
# track = multitrack.get_merged_pianoroll(mode='max')
# print(track.shape)
#
# pm = multitrack.to_pretty_midi()
# memFile = BytesIO()
# pm.write(memFile)
# memFile.seek(0)
# playMidiFile(memFile)



# pm = pretty_midi.PrettyMIDI(midi_path)
# #
# # fs = 100
# pr = get_piano_roll_mod(pm)
# print(pm.get_end_time())
# print(pr.shape)
# # pr = pr[0:200,:]
# pianoMD = piano_roll_to_pretty_midi(pr, program= 1)
# # print(np.amax(pr))
#
# out_name = folder_name + midi_name + '//' + 'piano' + '.mid'
# # for inst in pianoMD.instruments:
# #     print(inst)
# #     piano_roll = inst.get_piano_roll(fs=30)
# #     print(np.amax(piano_roll))
# # print(pianoMD.estimate_tempo())
# print("...")
# # print(pm.get_end_time())
# print(pianoMD.get_end_time())
#
# prr = get_piano_roll_mod(pianoMD)
# print(prr.shape[1] )
# memFile = BytesIO()
# pianoMD.write(memFile)
# memFile.seek(0)
# playMidiFile(memFile)
