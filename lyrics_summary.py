import pandas as pd
import os
from dotenv import load_dotenv
import re
import google.generativeai as genai
import google.api_core.exceptions

load_dotenv()


import lyricsgenius
import time


class LyricsSummary:
    def __init__(self, timeout=3):
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

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        # if lyrics.csv exists, load it
        if os.path.exists("lyrics.csv"):
            self.lyrics_df = pd.read_csv("lyrics.csv", sep=";")
        else:
            self.lyrics_df = pd.DataFrame(
                columns=["artist", "title", "release_date", "lyrics_summary"]
            )

    def get_lyrics(self, max_retries=5, wait_time=2):

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

                while retries < max_retries:
                    try:

                        response = self.model.generate_content(
                            "Here are some lyrics from the song "
                            + title
                            + " by "
                            + artist
                            + ". Write me a summary of the song. Dont try to interpret the lyrics, just summarize them. Dont speak about the lyrics being in a song and refer to the author as 'A person'. Take the lyrics and just assume that they are true. Do it with 5 sentances and only return the summarized text:\n"
                            + result,
                        )
                        retries = 0
                        wait_time = 2
                        break

                    except google.api_core.exceptions.ResourceExhausted:
                        retries += 1
                        print(
                            f"ResourceExhausted error for {title} by {artist}. Retrying in {wait_time} seconds"
                        )
                        time.sleep(wait_time)
                        wait_time *= 2
                        continue

                if retries == max_retries:
                    raise Exception("Max gemini retries reached")
                # add a new row to the dataframe
                new_row = pd.DataFrame(
                    [
                        {
                            "artist": artist,
                            "title": title,
                            "release_date": release_date,
                            "lyrics_summary": response.text,
                        }
                    ]
                )

                self.lyrics_df = pd.concat([self.lyrics_df, new_row], ignore_index=True)

                t2 = time.time()

                duration = t2 - t1
                print(
                    f"Lyrics for {title} by {artist} were processed in {duration} seconds"
                )

                if duration < self.timeout:
                    time.sleep(self.timeout - duration)

                if index % 10 == 0:
                    self.lyrics_df.to_csv("lyrics.csv", index=False, sep=";")


if __name__ == "__main__":
    lyrics_summary = LyricsSummary(timeout=3)
    lyrics_summary.get_lyrics()
