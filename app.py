from flask import Flask, jsonify, request
import wikipedia
import spacy
import yake
from transformers import AutoTokenizer, AutoModel
import numpy as np
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
import re
import requests
from bs4 import BeautifulSoup

env_path = os.path.join(os.getcwd(), ".env")
load_dotenv(env_path)

# print all files at the root of the project
cwd = os.getcwd()
print(f"FILES: {os.listdir(cwd)}")

api_key = os.environ.get("PINECONE_KEY")
lyrics_uid = os.environ.get("LYRICS_USER")
lyrics_token = os.environ.get("LYRICS_TOKEN")

if not api_key:
    raise ValueError("PINECONE_KEY environment variable not set")

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")
keyword_extractor = yake.KeywordExtractor(
    lan="en",  # Language
    n=1,  # Maximum size of the n-gram
    dedupLim=0.9,  # Deduplication threshold
    top=10,
    features=None,  # Use default features
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

pc = Pinecone(api_key=api_key)
index_name = "wiki-song-match"
index = pc.Index(index_name)
namespace = "lyrics_embedding"

nltk.download("stopwords")
nltk.download("punkt_tab")

# Load English stop words
stop_words = set(stopwords.words("english"))

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


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


def get_embedding(keyword):
    inputs = tokenizer(keyword, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy().flatten().tolist()


def embeddings_from_keywords(keywords, weights=None):
    if weights is None:
        weights = [x[1] for x in keywords]
        keywords = [x[0] for x in keywords]
    np_weights = np.array(weights)
    np_weights = np_weights / np.sum(np_weights)  # Normalize weights

    embeddings = np.array([get_embedding(k) for k in keywords])  # (10, 768)

    # Compute weighted sum
    combined_embedding = np.sum(
        np_weights[:, np.newaxis] * embeddings, axis=0
    )  # (768,)

    # Normalize the final embedding
    combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
    return combined_embedding


def extract_wiki_keywords(article):
    try:
        text = wikipedia.page(article).content
    except:
        return

    cleaned_text = clean_text(text)

    keywords = keyword_extractor.extract_keywords(cleaned_text)
    if len(keywords) < 10:
        return

    combined_embedding = embeddings_from_keywords(keywords)

    # Concatenate with the new dataframe
    return keywords, combined_embedding.tolist()


def get_lyrics(artist: str, song: str):
    artist = artist.replace(" ", "%20")
    song = song.replace(" ", "%20")
    url = f"https://www.stands4.com/services/v2/lyrics.php?uid={lyrics_uid}&tokenid={lyrics_token}&term={song}&artist={artist}&format=json"

    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "cache-control": "max-age=0",
        "priority": "u=0, i",
        "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed with status code: {response.status_code}")
        return None
    result = response.json().get("result")
    if not result:
        return None
    song_link = result[0].get("song-link")

    lyrics_response = requests.get(song_link, headers=headers)
    soup = BeautifulSoup(lyrics_response.text, "html.parser")
    lyrics_div = soup.find("pre", class_="lyric-body")
    if not lyrics_div:
        return None
    lyrics = lyrics_div.get_text(strip=True)
    return lyrics


@app.route("/match/<path:article>", defaults={"num": 1})
@app.route("/match/<path:article>/<int:num>")
def get_song_match(article, num):
    if num > 10:
        return (
            jsonify({"error": "Number of recommendations should be less than 10"}),
            400,
        )
    result = extract_wiki_keywords(article)
    if result is None:
        return jsonify({"error": "No wikipedia page found"}), 404

    keywords, embedding = result
    embedding = [float(x) for x in embedding]

    recommendation = index.query(
        vector=[embedding],
        top_k=num,
        namespace=namespace,
        include_metadata=True,
        include_values=True,
    )

    return_dict = {"data": []}

    for result in recommendation.matches:
        return_dict["data"].append(
            {
                "title": result.metadata["title"],
                "artist": result.metadata["artist"],
                "release_date": result.metadata["release_date"],
                "lyrics_keywords": result.metadata["keywords"],
                "lyrics_embedding": result.values,
                "score": result.score,
                "wiki_keywords": keywords,
                "wiki_embedding": embedding,
            }
        )

    return jsonify(return_dict), 200


@app.route("/lyrics_keywords", methods=["POST"])
def get_lyrics_keywords():
    data = request.get_json()
    if not data or "artist" not in data or "song" not in data:
        return jsonify({"error": "No artist or song provided"}), 400

    artist = data["artist"]
    song = data["song"]
    lyrics = get_lyrics(artist, song)

    if lyrics is None:
        return jsonify({"error": "Lyrics not found"}), 404

    cleaned_text = clean_text(lyrics)
    keywords = keyword_extractor.extract_keywords(cleaned_text)

    weights = [x[1] for x in keywords]
    keywords = [x[0] for x in keywords]

    if len(keywords) < 10:
        return jsonify({"error": "Not enough keywords found"}), 404

    embedding = embeddings_from_keywords(keywords, weights)
    embedding = [float(x) for x in embedding]

    return (
        jsonify({"embedding": embedding, "keywords": keywords, "weights": weights}),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True)
