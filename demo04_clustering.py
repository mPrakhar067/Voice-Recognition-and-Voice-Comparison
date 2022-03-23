from sklearn.linear_model import LogisticRegression 
from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path
from tqdm import tqdm
import numpy as np


## Gather a single utterance per speaker
data_dir = Path("audio_data", "librispeech_train-clean-100")
wav_fpaths = list(data_dir.glob("*.flac"))
speakers = [fpath.stem.split("-")[0] for fpath in wav_fpaths]
wavs = [preprocess_wav(wav_fpath) for wav_fpath in \
        tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit=" utterances")]

# Get the sex of each speaker from the metadata file
with data_dir.joinpath("SPEAKERS.TXT").open("r") as f:
    sexes = dict(l.replace(" ", "").split("|")[:2] for l in f if not l.startswith(";"))
markers = ["x" if sexes[speaker] == "M" else "o" for speaker in speakers]
colors = ["black"] * len(speakers)


## Compute the embeddings
encoder = VoiceEncoder()
utterance_embeds = np.array(list(map(encoder.embed_utterance, wavs)))


## Project the embeddings in 2D space. 
_, ax = plt.subplots(figsize=(6, 6))
# Passing min_dist=1 to UMAP will make it so the projections don't necessarily need to fit in 
# clusters, so that you can have a better idea of what the manifold really looks like. 
projs = plot_projections(utterance_embeds, speakers, ax, colors, markers, False,
                         min_dist=1)
ax.set_title("Embeddings for %d speakers" % (len(speakers)))
ax.scatter([], [], marker="x", c="black", label="Male speaker")
ax.scatter([], [], marker="o", c="black", label="Female speaker")

# Separate the data by the sex
classifier = LogisticRegression(solver="lbfgs")
classifier.fit(projs, markers)
x = np.linspace(*ax.get_xlim(), num=200)
y = -(classifier.coef_[0, 0] * x + classifier.intercept_) / classifier.coef_[0, 1]
mask = (y > ax.get_ylim()[0]) & (y < ax.get_ylim()[1])
ax.plot(x[mask], y[mask], label="Decision boundary")

ax.legend()
plt.show()
