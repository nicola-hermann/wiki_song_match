from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


def get_keyword_embeddings(keywords):
    """
    Get embeddings for a list of keywords using a pretrained BERT model.

    Args:
        keywords (list): List of keywords.

    Returns:
        torch.Tensor: A tensor of embeddings for the keywords (shape: len(keywords) x hidden_size).
    """
    if not keywords:
        return torch.empty(0)  # Return an empty tensor if no keywords are provided

    inputs = tokenizer(keywords, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token's embedding as the representation for each keyword
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings


wiki_df = pd.read_csv("wiki_keywords.csv", sep=";")
lyrics_df = pd.read_csv("keywords.csv", sep=";")

wiki_dict = {}

for index, row in tqdm(wiki_df.iterrows()):
    title = row["article"]
    keywords = row["keywords"]
    embeddings = get_keyword_embeddings(keywords)
    embeddings = embeddings.mean(dim=0)
    wiki_dict[title] = embeddings

lyrics_dict = {}

for index, row in tqdm(lyrics_df.iterrows()):
    title = row["title"]
    artist = row["artist"]
    keywords = row["keywords"]
    embeddings = get_keyword_embeddings(keywords)
    embeddings = embeddings.mean(dim=0)
    lyrics_dict[f"{artist}_{title}"] = embeddings

wiki_embeddings = np.array(list(wiki_dict.values()))
song_embeddings = np.array(list(lyrics_dict.values()))

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
