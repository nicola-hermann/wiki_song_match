from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import pandas as pd
from tqdm import tqdm
import time
import itertools

load_dotenv()


def chunks(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


api_key = os.environ.get("PINECONE_KEY")
pc = Pinecone(api_key=api_key)
index_name = "wiki-song-match"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)


lyrics_df = pd.read_feather("data\lyrics_embeddings.feather")
# reset index
lyrics_df.reset_index(inplace=True)
lyrics_list = []

for idx, row in lyrics_df.iterrows():
    lyrics_list.append(
        {
            "id": str(idx),
            "values": row["embedding"].tolist(),
            "metadata": {
                "title": row["track_name"],
                "artist": row["artist_name"],
                "release_date": row["release_date"],
                "keywords": row["keywords"].tolist(),
            },
        }
    )

index = pc.Index(index_name)
batch_size = 500

for ids_vectors_chunk in chunks(lyrics_list, batch_size=200):
    index.upsert(vectors=ids_vectors_chunk, namespace="lyrics_embedding")

time.sleep(10)
print(index.describe_index_stats())
print("Successfully loaded embeddings to Pinecone!")
