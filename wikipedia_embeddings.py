import wikipedia
from sentence_transformers import SentenceTransformer
import pysbd
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

model = SentenceTransformer("all-MiniLM-L6-v2")
segmenter = pysbd.Segmenter(language="en", clean=True)

wiki_embeddings_dict = {}


# get a random page from wikipedia
for i in tqdm(range(100)):
    article = wikipedia.random()
    # Get a summary of the article
    try:
        summary = wikipedia.summary(article)
    except Exception as e:
        continue

    sentences = segmenter.segment(summary)
    # get the fist 5 sentences
    sentences = sentences[:5]

    # get the embeddings
    embeddings = model.encode(sentences)

    wiki_average = np.mean(embeddings, axis=0)

    # store the embeddings
    wiki_embeddings_dict[article] = wiki_average

song_embeddings_dict = {}
songs_df = pd.read_csv("lyrics.csv", sep=";")

for index, row in tqdm(songs_df.iterrows()):
    if index == 100:
        break
    title = row["title"]
    artist = row["artist"]
    lyrics_summary = row["lyrics_summary"]

    sentences = segmenter.segment(lyrics_summary)
    sentences = sentences[:5]

    embeddings = model.encode(sentences)
    lyrics_average = np.mean(embeddings, axis=0)

    song_embeddings_dict[f"{title}_{artist}"] = lyrics_average


wiki_embeddings = np.array(list(wiki_embeddings_dict.values()))
song_embeddings = np.array(list(song_embeddings_dict.values()))

all_embeddings = np.vstack([wiki_embeddings, song_embeddings])
labels = ["Wikipedia"] * len(wiki_embeddings) + ["Lyrics"] * len(song_embeddings)

# Reduce dimensions
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(all_embeddings)

# Plot
plt.figure(figsize=(10, 8))
for label, color in zip(["Wikipedia", "Lyrics"], ["blue", "green"]):
    indices = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(
        reduced_embeddings[indices, 0],
        reduced_embeddings[indices, 1],
        label=label,
        alpha=0.6,
    )

plt.legend()
plt.title("t-SNE Visualization of Embedding Spaces")
plt.show()
