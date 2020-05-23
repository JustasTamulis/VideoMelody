# VideoMelody

To install requirements run:

>> pip install -r requirements.txt

This also requires ffmpeg. https://ffmpeg.org/


To create a melody, simply run

>> python Melodize.py input_video_path

Avi, mp4 are supported, many other formats will probably work as well.

An enocder is used which may not work well if the video presented is very different from its training dataset. Consider providing outdoors video with wide frame of nature or city views.

A result will be generated in project folder.




Moreover, steps to utilize most of the system is provided, with main parts being:

1) Dataset creation
2) Training
3) Evaluating


1) The preparation of the training data, is not user friendly. A mix of absolute and relative paths is used. A guidelines are given, but a dataset of MIDI and video files has to be provided.

i) Resize video to 120 x 120. Train autoencoder with 'Autoencoder/train.py' providing the paths to the dataset. Use 'make_video_vec.py' to encode the video.
ii) Use 'PreMidi.py' for the MIDI files, run through them one time to binarize them. Use 'ohe.py' to create one hot encoded represantions, they are saved as numpy arrays.
iii) In the Data file, use 'allign.py' to adjust increase frames of the video. Use 'data_prep.py' to create small arrays that can be fed to the networks. It already prepares the data needed for Models 1-4.
iv) Adjust the 'feeder.py' which is uitilized in the training process to obtain the data.


2) The training process can be run, with:

>> wandb run python train_model.py

Wandb needs to be initialized to the current folder. Please refer to https://docs.wandb.com/quickstart. The Model architecture can be selected from 1 to 4.


3) To evaluate results:

i) run 'make_tests', to create MIDI files for the model selected.
ii) Run 'metrics.py' to evaluate with selected metrics.
