To install requirements run:

>> pip install -r requirements.txt


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

Wandb needs to be initialized to the current folder. Please refer to https://docs.wandb.com/quickstart.



3) To evaluate results:

i) run 'make_tests', to create MIDI files for the model selected.
ii) Run 'metrics.py' to evaluate with selected metrics.