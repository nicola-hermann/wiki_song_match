from flask import Flask, jsonify
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

env_path = os.path.join(os.getcwd(), ".env")
load_dotenv(env_path)

# print all files at the root of the project
cwd = os.getcwd()
print(f"FILES: {os.listdir(cwd)}")

api_key = os.environ.get("PINECONE_KEY")
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
    return (
        outputs.last_hidden_state[:, 0, :].detach().numpy().flatten().tolist()
    )  # Use [CLS] token embedding


def extract_wiki_keywords(article):
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
    np_weights = np.array(weights)
    np_weights = np_weights / np.sum(np_weights)  # Normalize weights

    embeddings = np.array([get_embedding(k) for k in keywords])  # (10, 768)

    # Compute weighted sum
    combined_embedding = np.sum(
        np_weights[:, np.newaxis] * embeddings, axis=0
    )  # (768,)

    # Normalize the final embedding
    combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

    # Concatenate with the new dataframe
    return keywords, np_weights.tolist(), combined_embedding.tolist()


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

    keywords, _, embedding = result
    embedding = [float(x) for x in embedding]

    recommendation = index.query(
        vector=[embedding],
        top_k=num,
        namespace=namespace,
        include_metadata=True,
        include_values=False,
    )

    return_dict = {"data": []}
    for result in recommendation.matches:
        return_dict["data"].append(
            {
                "title": result.metadata["title"],
                "artist": result.metadata["artist"],
                "release_date": result.metadata["release_date"],
                "lyrics_keywords": result.metadata["keywords"],
                "score": result.score,
                "wiki_keywords": keywords,
            }
        )

    return jsonify(return_dict), 200


if __name__ == "__main__":
    app.run(debug=True)
