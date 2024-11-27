import pandas as pd
import os
from dotenv import load_dotenv
import re
import yake

load_dotenv()


import lyricsgenius
import time
import spacy


class LyricsKeywords:
    def __init__(self, timeout=1):
        self.timeout = timeout
        token = os.getenv("GENIUS_API_KEY")
        self.genius = lyricsgenius.Genius(token)

        self.genius.verbose = False
        self.genius.remove_section_headers = (
            True  # Remove section headers (e.g. [Chorus]) from lyrics when searching
        )
        self.genius.skip_non_songs = (
            True  # Include hits thought to be non-songs (e.g. track lists)
        )
        self.genius.excluded_terms = [
            "(Remix)",
            "(Live)",
        ]  # Exclude songs with these words in their title

        self.songs = pd.read_csv("tcc_ceds_music.csv")

        self.songs = self.songs.sample(frac=1).reset_index(drop=True)

        self.keyword_extractor = yake.KeywordExtractor(
            lan="en",  # Language
            n=1,  # Maximum size of the n-gram
            dedupLim=0.9,  # Deduplication threshold
            top=10,
            features=None,  # Use default features
        )
        self.nlp = spacy.load("en_core_web_sm")

        # if lyrics.csv exists, load it
        if os.path.exists("keywords.csv"):
            self.lyrics_df = pd.read_csv("keywords.csv", sep=";")
        else:
            self.lyrics_df = pd.DataFrame(
                columns=["artist", "title", "release_date", "keywords"]
            )

    def lemmatize_with_spacy(self, text):
        doc = self.nlp(text)
        return [token.lemma_ for token in doc]

    def get_lyrics_keywords(self, max_retries=5, wait_time=2):

        # iterate over the rows
        for index, row in self.songs.iterrows():
            t1 = time.time()
            artist = row["artist_name"]
            title = row["track_name"]
            release_date = row["release_date"]

            # Check if the song is already in the dataframe
            if not self.lyrics_df[
                (self.lyrics_df["artist"] == artist)
                & (self.lyrics_df["title"] == title)
            ].empty:
                print(f"Lyrics for {title} by {artist} already processed")
                continue
            retries = 0
            while retries < max_retries:
                try:
                    song = self.genius.search_song(title, artist)
                    retries = 0
                    wait_time = 2
                    break

                except Exception as e:
                    retries += 1
                    print(
                        f"Genius Error for {title} by {artist}. Retrying in {wait_time} seconds"
                    )
                    time.sleep(wait_time)
                    wait_time *= 2
                    continue

            if retries == max_retries:
                raise Exception("Max genius retries reached")

            # if the song is found
            if song:
                lyrics = song.lyrics
                if len(lyrics) > 10000:
                    print(
                        f"Lyrics for {title} by {artist} are too long, something went wrong"
                    )
                    continue
                # The lyrics start after the first occurrence of the word "Lyrics"
                lyrics = lyrics[lyrics.find("Lyrics") + len("Lyrics") :]
                # remove the embed word at the end
                lyrics = lyrics[: lyrics.find("Embed")]

                result = re.sub(r"\[.*?\]", "", lyrics).strip()

                lemmatized = self.lemmatize_with_spacy(result)

                lemmatized = " ".join(lemmatized)
                # replace the new lines with spaces
                lemmatized = lemmatized.replace("\n", " ")
                # remove ( and ) from the text
                lemmatized = lemmatized.replace("(", "").replace(")", "")

                keywords = self.keyword_extractor.extract_keywords(lemmatized)

                keywords_res = [keyword for keyword, _ in keywords]

                keywords_df = pd.DataFrame(
                    {
                        "artist": [artist],
                        "title": [title],
                        "release_date": [release_date],
                        "keywords": [keywords_res],
                    }
                )

                self.lyrics_df = pd.concat([self.lyrics_df, keywords_df])

                duration = time.time() - t1
                if duration < self.timeout:
                    time.sleep(self.timeout - duration)

                if index % 10 == 0:
                    self.lyrics_df.to_csv("keywords.csv", index=False, sep=";")


if __name__ == "__main__":
    lyrics_summary = LyricsKeywords()
    lyrics_summary.get_lyrics_keywords()
