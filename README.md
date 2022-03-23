Resemblyzer allows you to derive a **high-level representation of a voice** through a deep learning model (referred to as the voice encoder). Given an audio file of speech, it creates a summary vector of 256 values (an embedding, often shortened to "embed" in this repo) that summarizes the characteristics of the voice spoken. 

N.B.: this repo holds approx 100mb of audio data for demonstration purpose. To get the package alone, run `pip install resemblyzer` (python 3.5+ is required).

## Demos


**Cross-similarity**: [\[Demo 01\]](https://github.com/mPrakhar067/Voice-Recognition-and-Voice-Comparison/blob/master/cross_similarity.py) comparing 10 utterances from 10 speakers against 10 other utterances from the same speakers.

![demo_01](plots/sim_matrix_1.png?raw=true)


**Fake speech detection**: [\[Demo 02\]](https://github.com/mPrakhar067/Voice-Recognition-and-Voice-Comparison/blob/master/fake_speech_detection.py) modest detection of fake speech by comparing the similarity of 12 unknown utterances (6 real ones, 6 fakes) against ground truth reference audio. Scores above the dashed line are predicted as real, so the model makes one error here.

![demo_02](plots/fake_speech_detection01.png?raw=true)


## What can I do with this package?
Resemblyzer has many uses:
- **Voice similarity metric**: compare different voices and get a value on how similar they sound. This leads to other applications:
  - **Speaker verification**: create a voice profile for a person from a few seconds of speech (5s - 30s) and compare it to that of new audio. Reject similarity scores below a threshold.
  - **Speaker diarization**: figure out who is talking when by comparing voice profiles with the continuous embedding of a multispeaker speech segment.
  - **Fake speech detection**: verify if some speech is legitimate or fake by comparing the similarity of possible fake speech to real speech.
- **High-level feature extraction**: you can use the embeddings generated as feature vectors for machine learning or data analysis. This also leads to other applications:
  - **Voice cloning**:
  - **Component analysis**: figure out accents, tones, prosody, gender through a component analysis of the embeddings.
  - **Virtual voices**: create entirely new voice embeddings by sampling from a prior distribution.
- **Loss function**: you can backpropagate through the voice encoder model and use it as a perceptual loss for your deep learning model! The voice encoder is written in PyTorch.

Resemblyzer is fast to execute (around 1000x real-time on a GTX 1080, with a minimum of 10ms for I/O operations), and can run both on CPU or GPU. It is robust to noise. It currently works best on English language only, but should still be able to perform somewhat decently on other languages.


## Code example
This is a short example showing how to use Resemblyzer:
```python
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

fpath = Path("give_the_path_to_an_audio_file")
wav = preprocess_wav(fpath)

encoder = VoiceEncoder()
embed = encoder.embed_utterance(wav)
np.set_printoptions(precision=3, suppress=True)
print(embed)
```

