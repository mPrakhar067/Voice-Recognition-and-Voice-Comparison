from resemblyzer.hparams import *
from resemblyzer import audio
from pathlib import Path
from typing import Union, List
from torch import nn
from time import perf_counter as timer
import numpy as np
import torch


class VoiceEncoder(nn.Module):
    def __init__(self, device: Union[str, torch.device]=None, verbose=True, weights_fpath: Union[Path, str]=None):
       
        super().__init__()

        # Define the network
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

        # Get the target device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Load the pretrained model'speaker weights
        if weights_fpath is None:
            weights_fpath = Path(__file__).resolve().parent.joinpath("pretrained.pt")
        else:
            weights_fpath = Path(weights_fpath)

        if not weights_fpath.exists():
            raise Exception("Couldn't find the voice encoder pretrained model at %s." %
                            weights_fpath)
        start = timer()
        checkpoint = torch.load(weights_fpath, map_location="cpu")
        self.load_state_dict(checkpoint["model_state"], strict=False)
        self.to(device)

        if verbose:
            print("Loaded the voice encoder model on %s in %.2f seconds." %
                  (device.type, timer() - start))

    def forward(self, mels: torch.FloatTensor):
        # Pass the input through the LSTM layers and retrieve the final hidden state of the last
        # layer. Apply a cutoff to 0 for negative values and L2 normalize the embeddings.
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    @staticmethod
    def compute_partial_slices(n_samples: int, rate, min_coverage):
        assert 0 < min_coverage <= 1

        # Compute how many frames separate two partial utterances
        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round((sampling_rate / rate) / samples_per_frame))
        assert 0 < frame_step, "The rate is too high"
        assert frame_step <= partials_n_frames, "The rate is too low, it should be %f at least" % \
            (sampling_rate / (samples_per_frame * partials_n_frames))

        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

    def embed_utterance(self, wav: np.ndarray, return_partials=False, rate=1.3, min_coverage=0.75):
       
        # Compute where to split the utterance into partials and pad the waveform with zeros if
        # the partial utterances cover a larger range.
        wav_slices, mel_slices = self.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        # Split the utterance into partials and forward them through the model
        mel = audio.wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        with torch.no_grad():
            mels = torch.from_numpy(mels).to(self.device)
            partial_embeds = self(mels).cpu().numpy()

        # Compute the utterance embedding from the partial embeddings
        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)

        if return_partials:
            return embed, partial_embeds, wav_slices
        return embed

    def embed_speaker(self, wavs: List[np.ndarray], **kwargs):
       
        raw_embed = np.mean([self.embed_utterance(wav, return_partials=False, **kwargs) \
                             for wav in wavs], axis=0)
        return raw_embed / np.linalg.norm(raw_embed, 2)
