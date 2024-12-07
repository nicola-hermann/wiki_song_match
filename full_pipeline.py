import pandas as pd
import yake
import wikipedia

mode = "pc"  # or "local"

from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import re
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.neighbors import NearestNeighbors
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

if mode == "local":
    lyrics_df = pd.read_feather("data\lyrics_embeddings.feather")
    lyrics_embeddings = lyrics_df["embedding"].values
    lyrics_embeddings = np.vstack(lyrics_embeddings)

    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(lyrics_embeddings)

else:
    from pinecone import Pinecone

    api_key = os.environ.get("PINECONE_KEY")
    pc = Pinecone(api_key=api_key)
    index_name = "wiki-song-match"
    index = pc.Index(index_name)
    namespace = "lyrics_embedding"


# Download necessary resources
nltk.download("stopwords")
nltk.download("punkt_tab")

# Load English stop words
stop_words = set(stopwords.words("english"))

# Load the spaCy language model for lemmatization
nlp = spacy.load("en_core_web_sm")


def clean_text(text):
    """
    Cleans text by:
    1. Removing symbols and numbers
    2. Removing stop words
    3. Lemmatizing to canonical form
    """
    # Step 1: Remove symbols and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only alphabets
    text = text.lower()  # Convert to lowercase

    # Step 2: Tokenize and remove stop words
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    # Step 3: Lemmatize the remaining words
    lemmatized_tokens = []
    for token in tokens:
        doc = nlp(token)
        lemmatized_tokens.append(doc[0].lemma_)

    # Join tokens back into a string
    cleaned_text = " ".join(lemmatized_tokens)

    return cleaned_text


# Initialize the model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


# Function to get embeddings for a single keyword
def get_embedding(keyword):
    inputs = tokenizer(keyword, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding


def main(article):
    try:
        text = wikipedia.page(article).content
    except:
        return

    cleaned_text = clean_text(text)

    kw_extractor = yake.KeywordExtractor(
        lan="en", n=1, dedupLim=0.9, windowsSize=1, top=10
    )

    keywords = kw_extractor.extract_keywords(cleaned_text)
    weights = [x[1] for x in keywords]
    if len(keywords) < 10:
        return

    keywords = [x[0] for x in keywords]
    torch_weights = F.normalize(torch.tensor(weights), p=1, dim=0).unsqueeze(1)

    embeddings = torch.cat([get_embedding(k) for k in keywords], dim=0)  # (10, 768)

    # Compute weighted sum
    combined_embedding = torch.sum(torch_weights * embeddings, dim=0)  # (768,)

    # Normalize the final embedding
    combined_embedding = F.normalize(combined_embedding, p=2, dim=0).tolist()

    # Concatenate with the new dataframe
    return keywords, torch_weights.tolist(), combined_embedding


aritcle = "World War II"
result = main(aritcle)
if result is not None:
    keywords, weights, combined_embedding = main(aritcle)
    if mode == "local":
        # Find the nearest neighbors
        query_embedding = np.array([combined_embedding])
        distances, indices = knn.kneighbors(query_embedding)

        # Get the metadata for the nearest neighbors
        nearest_neighbors = lyrics_df.iloc[indices[0]]
        print(nearest_neighbors)
        # print(nearest_neighbors["artist_name", "track_name", "release_date", "keywords"])
        print(distances[0])
        breakpoint()
    else:

        combined_embedding = [float(value) for value in combined_embedding]

        results = index.query(
            vector=[combined_embedding],
            top_k=5,
            namespace=namespace,
            include_values=False,
            include_metrics=True,
            include_metadata=True,
        )

        return_dict = {"data": []}
        for result in results.matches:
            return_dict["data"].append(
                {
                    "title": result.metadata["title"],
                    "artist": result.metadata["artist"],
                    "release_date": result.metadata["release_date"],
                    "keywords": result.metadata["keywords"],
                    "score": result.score,
                }
            )
        print(return_dict)
else:
    print("Error: No keywords extracted")
