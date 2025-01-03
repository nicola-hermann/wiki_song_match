# Installation 

There are 2 "sub-repos" inside this repository that you can install:

- Full pipeline (Creating your own Vector Databse)
- Guessing game (Find a better song than there is in the database)

## Full Pipeline

There are alot of ways to install a repository. The following instructions are testen on Windows and MacOS.

1) Install python: This repository was built and tested on python 3.11.10

2) Create virtual environment and install requirements:

**Windows**

    
    python -m venv .venv
    ".venv/Scripts/activate"
    pip install -r requirements_full.txt


**MacOS**

    
    python -m venv .venv
    source .venv/bin/activate
    pip insatll -r requirements_full.txt
    

3) Download dataset from [here.](https://data.mendeley.com/datasets/3t9vbwxgr5/3) Save it as `dataset.csv` and put it inside the data folder. Otherwise, adapt the path inside `load_embeddings_to_pineconde.py` 

4) Create an account in [Pinecone](https://www.pinecone.io/) and create an [API key](https://docs.pinecone.io/guides/projects/manage-api-keys).

5) Create a `.env` file and add your API key inside the file. It should look like this:

    PINECONE_KEY=enterYourKeyHere

    
6) Load your data inside the vector database by running:
    
    python scripts/generate_lyrics_embeddings.py
    python scripts/load_embeddings_to_pinecone.py

**This process can take several hours!**

Note: the first script can be interruped and resumed any time. Progress is saved every 100 steps.

7) Try the matching by running: `python get_match.py -a <your-article>`. If you don't pass an article, a random wikipedia article is taken.

8) Optional: Create your own TSNE plot:


    python scripts/generate_wikipedia_embeddings.py
    python scripts/create_tsne_plot.py


**This process can take several hours!**

Note: the first script can be interruped and resumed any time. Progress is saved every 100 steps.

9) Optional: Host your own API

To host your own API with flask, you also need an API key for [Lyrics.com](https://www.lyrics.com/lyrics_api.php). Apply and wait for their email. In your .env append 2 more variables:



    LYRICS_USER=<YourUserID>
    LYRICS_TOKEN=<YourToken>

10) Optional: Deploy your own API on the cloud

To deploy the flask API on the Google Cloud, follow [this Tutorial](https://lesliemwubbel.com/setting-up-a-flask-app-and-deploying-it-via-google-cloud/)



## Guessing Game

*If you already did the Full Pipeline installation, skip to step 3.*

There are alot of ways to install a repository. The following instructions are testen on Windows and MacOS.

1) Install python: This repository was built and tested on python 3.11.10

2) Create virtual environment and install requirements:

**Windows**

    
    python -m venv .venv
    ".venv/Scripts/activate"
    pip install -r requirements_game.txt


**MacOS**

    
    python -m venv .venv
    source .venv/bin/activate
    pip insatll -r requirements_game.txt

3) Run the game with `python game.py`
    
