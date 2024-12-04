from flask import Flask
import wikipedia
import spacy
import yake
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import numpy as np

# from sklearn.neighbors import NearestNeighbors
# from sklearn.manifold import TSNE
# import pandas as pd
# import plotly.express as px

load_dotenv()
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

api_key = os.environ.get("PINECONE_KEY")

pc = Pinecone(api_key=api_key)
index_name = "wiki-song-match"
index = pc.Index(index_name)

# lyrics_list = np.load("data/lyrics_list.npy", allow_pickle=True)
# lyrics_embeddings = np.array([lyrics["values"] for lyrics in lyrics_list])
# metadata = [lyrics["metadata"] for lyrics in lyrics_list]

# knn = NearestNeighbors(n_neighbors=5)
# knn.fit(lyrics_embeddings)
# tsne = TSNE(n_components=2, random_state=42)


def lemmatize_with_spacy(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]


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


@app.route("/<path:article>")
def get_song_match(article):
    try:
        full_article = wikipedia.page(article)
    except Exception as e:
        return f"Error: {e}"

    lemmatized_article = lemmatize_with_spacy(full_article.content)
    keywords = keyword_extractor.extract_keywords(" ".join(lemmatized_article))
    keywords_res = [keyword for keyword, _ in keywords]
    breakpoint()
    embeddings = get_keyword_embeddings(keywords_res)

    embeddings_1dim = embeddings.mean(dim=0)

    embedding_normalized = embeddings_1dim / np.linalg.norm(embeddings_1dim)
    # embedding_reshaped = embedding_normalized.reshape(1, -1)

    # distances, indices = knn.kneighbors(embedding_reshaped)

    # response = [metadata[index] for index in indices[0]]

    # create a TSNE plot with the lyrics embeddings in one color and the wikipedia embeddings in another color
    # all_embeddings = np.vstack([lyrics_embeddings, embedding_reshaped])
    # labels = ["Lyrics"] * len(lyrics_embeddings) + ["Wikipedia"]
    # names = np.array(
    #     [lyrics["metadata"]["title"] for lyrics in lyrics_list] + [article]
    # )

    # embeddings_2d = tsne.fit_transform(all_embeddings)

    # df = pd.DataFrame(
    #     {
    #         "TSNE-1": embeddings_2d[:, 0],
    #         "TSNE-2": embeddings_2d[:, 1],
    #         "Domain": labels,
    #         "Name": names,
    #     }
    # )

    # # Plot with Plotly, showing names on hover
    # fig = px.scatter(
    #     df,
    #     x="TSNE-1",
    #     y="TSNE-2",
    #     color="Domain",
    #     title="t-SNE Visualization of Embeddings",
    #     labels={"color": "Domain"},
    #     hover_data={"Name": True, "TSNE-1": False, "TSNE-2": False},
    # )  # Show only 'Name' on hover
    # fig.update_traces(marker=dict(size=5, opacity=0.8))
    # fig.update_layout(legend_title_text="Domains")

    # # Show plot
    # fig.show()
    # breakpoint()

    recommendations = index.query(
        namespace="lyrics",
        vector=embedding_normalized.tolist(),
        top_k=3,
        include_metadata=True,
        include_values=False,
    )

    response = dict(recommendations["matches"][0]["metadata"])

    return response


if __name__ == "__main__":
    app.run(debug=True)
