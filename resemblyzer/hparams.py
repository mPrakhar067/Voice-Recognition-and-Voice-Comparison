
## Mel-filterbank
mel_window_length = 25  # In milliseconds
mel_window_step = 10    # In milliseconds
mel_n_channels = 40


## Audio
sampling_rate = 16000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 160     # 1600 ms


## Voice Activation Detection
vad_window_length = 30  # In milliseconds
 
vad_moving_average_width = 8
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6


## Audio volume normalization
audio_norm_target_dBFS = -30


## Model parameters
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3

