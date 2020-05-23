import subprocess
import os
import time

folder_name = "F:\\CompSci\\project\\MIDI\\DaftPunk\\"
# midi_name = 'OneMoreTime'
midi_name = 'HarderBetterFasterStronger'
# midi_name = 'Voyager'
mp4_path = folder_name + midi_name + '//' + midi_name + '.mp4'
midi_path = folder_name + midi_name + '//' + midi_name + '.mid'
wav_path = folder_name + midi_name + '//' + midi_name + '.wav'

command = "ffmpeg -loglevel panic -i " + mp4_path + " -ab 160k -ac 2 -ar 44100 -vn " + wav_path

subprocess.call(command, shell=True)

while os.path.isfile(wav_path) == False:
    time.sleep(1)
print('Converted to Wav')
#
gen_path = folder_name + midi_name + '//' + 'generated' + '.mid'
command = "audio-to-midi "+wav_path+" --output " + gen_path

subprocess.call(command, shell=True)
