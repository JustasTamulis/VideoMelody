
def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))

plt.figure(figsize=(12, 4))
plot_piano_roll(pm, 24, 84)
input("exit")
input("Press Enter to continue...")
if input("Press enter to play, anything else to exit") == '':
    print("Movie is starting soon")
    movie.play()

input("Press Enter to cut...")
movie.cut(40,400)
input("CUT !")
input("Press Enter to continue...")
if input("Press enter to play, anything else to exit") == '':
    print("Movie is starting soon")
    movie.play_cut()



pygame.mixer.pre_init(44100, 16, 2, 4096)
print(pm.get_pitch_class_histogram())
pygame.mixer.init()
pygame.mixer.music.load(folder_name + midi_name + '.mid')

print(pm.estimate_tempo())
# Compute the relative amount of each semitone across the entire song, a proxy for key
total_velocity = sum(sum(pm.get_chroma()))
print([sum(semitone)/total_velocity for semitone in pm.get_chroma()])
# Shift all notes up by 5 semitones
for instrument in pm.instruments:
    # Don't want to shift drum notes
    if not instrument.is_drum:
        for note in instrument.notes:
            note.pitch += 5
# Synthesize the resulting MIDI data using sine waves
audio_data = pm.synthesize()
print(pygame.mixer.get_init())
pygame.mixer.music.play(20, 0.0)
pygame.mixer.music.load(pm)
s = pygame.mixer.Sound('F:/CompSci/project/MIDI/' + midi_name + '.mid')
s.play()
while pygame.mixer.music.get_busy():
    pygame.time.wait(1000)
pygame.time.wait(10000)
f = open("demofile.txt", "r")
print(f.read(5))

dict_time_notes = {}
m = {}
for inst in pm.instruments:

    inst.program = 1
    print(inst)
    piano_roll = inst.get_piano_roll(fs=30)
    dict_time_notes = piano_roll
    sample = dict_time_notes
    times = np.unique(np.where(sample > 0)[1])
    index = np.where(sample > 0)
    dict_keys_time = {}

    for time in times:
        index_where = np.where(index[1] == time)
        notes = index[0][index_where]
        dict_keys_time[time] = notes
    music = dict_keys_time
    # for sec in music.values():
        # input(sec)
    m = music
    input("next")
for inst in pm.instruments:
    print(inst)
print("...............")
pm.instruments.pop(0)
pm.instruments.pop(3)
for inst in pm.instruments:
    print(inst)
    piano_roll = inst.get_piano_roll(fs=30)
    print(np.amax(piano_roll))
