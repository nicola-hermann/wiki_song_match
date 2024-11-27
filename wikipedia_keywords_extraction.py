import wikipedia
from tqdm import tqdm
import spacy
import yake
import os
import pandas as pd


wiki_embeddings_dict = {}
nlp = spacy.load("en_core_web_sm")
keyword_extractor = yake.KeywordExtractor(
    lan="en",  # Language
    n=1,  # Maximum size of the n-gram
    dedupLim=0.9,  # Deduplication threshold
    top=10,
    features=None,  # Use default features
)


def lemmatize_with_spacy(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]


if os.path.exists("wiki_keywords.csv"):
    wiki_df = pd.read_csv("wiki_keywords.csv", sep=";")
else:
    wiki_df = pd.DataFrame(columns=["article", "keywords"])


# get a random page from wikipedia
for i in tqdm(range(2000)):
    # get the full article of the page
    article = wikipedia.random()
    try:
        full_article = wikipedia.page(article)
    except Exception as e:
        continue

    # check if the article is already in the dataframe
    if not wiki_df[wiki_df["article"] == article].empty:
        print(f"Keywords for {article} already processed")
        continue

    # lemmatize the article
    lemmatized_article = lemmatize_with_spacy(full_article.content)
    # get the keywords
    keywords = keyword_extractor.extract_keywords(" ".join(lemmatized_article))
    keywords_res = [keyword for keyword, _ in keywords]

    article_df = pd.DataFrame({"article": [article], "keywords": [keywords_res]})
    wiki_df = pd.concat([wiki_df, article_df])

    if i % 10 == 0:
        wiki_df.to_csv("wiki_keywords.csv", sep=";", index=False)
