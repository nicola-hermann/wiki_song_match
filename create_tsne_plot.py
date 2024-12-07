import pandas as pd

from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np

# Load the embeddings
lyrics_df = pd.read_feather("data\lyrics_embeddings.feather")
wiki_df = pd.read_feather("data\wiki_embeddings.feather")

lyrics_embeddings = np.vstack(lyrics_df["embedding"].values)
wiki_embeddings = np.vstack(wiki_df["embedding"].values)

# Perform t-SNE on the embeddings
tsne = TSNE(n_components=2, random_state=42)
stacked_embeddings = np.vstack([lyrics_embeddings, wiki_embeddings])
stacked_color = ["lyrics"] * len(lyrics_df) + ["wiki"] * len(wiki_df)
stacked_names = lyrics_df["track_name"].tolist() + wiki_df["article"].tolist()

tsne_embeddings = tsne.fit_transform(stacked_embeddings)

# Create a plot
tsne_df = pd.DataFrame(
    {
        "x": tsne_embeddings[:, 0],
        "y": tsne_embeddings[:, 1],
        "color": stacked_color,
        "name": stacked_names,
    }
)

fig = px.scatter(
    tsne_df, x="x", y="y", color="color", hover_name="name", title="t-SNE Plot"
)

fig.show()

fig.write_html("tsne_plot.html")
