from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm
import time
import numpy as np

load_dotenv()

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


def batch_data(data, batch_size):
    """
    Split data into smaller batches.

    Args:
        data (list): List of vectors to upsert.
        batch_size (int): Maximum number of vectors per batch.

    Returns:
        list: A list of batches, where each batch is a subset of the data.
    """
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


api_key = os.environ.get("PINECONE_KEY")

pc = Pinecone(api_key=api_key)
index_name = "wiki-song-match"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,  # Replace with your model dimensions
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)

if not os.path.exists("data/lyrics_list.npy"):

    lyrics_df = pd.read_csv("data/keywords.csv", sep=";")
    lyrics_list = []

    for idx, row in tqdm(lyrics_df.iterrows()):
        title = row["title"]
        artist = row["artist"]
        release_date = row["release_date"]
        keywords = row["keywords"]
        if len(keywords) < 10:
            print(f"Skipping {title} by {artist} due to insufficient keywords")
            continue
        embeddings = get_keyword_embeddings(keywords)
        embedding_1d = embeddings.mean(dim=0)
        # normalize the embeddings
        embedding_normalized = embedding_1d / torch.linalg.norm(embedding_1d)
        embeddings_list = embedding_normalized.tolist()

        lyrics_list.append(
            {
                "id": str(idx),
                "values": embeddings_list,
                "metadata": {
                    "title": title,
                    "artist": artist,
                    "release_date": release_date,
                },
            }
        )

    np.save("lyrics_list.npy", lyrics_list)
else:
    print("Skipping embedding generation, loading existing data")
    lyrics_list = np.load("data/lyrics_list.npy", allow_pickle=True)

index = pc.Index(index_name)
batch_size = 500
for batch in tqdm(batch_data(lyrics_list, batch_size)):

    index.upsert(vectors=batch, namespace="lyrics")

time.sleep(10)  # Wait for the upserted vectors to be indexed

print(index.describe_index_stats())
