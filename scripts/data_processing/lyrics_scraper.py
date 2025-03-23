import requests
from bs4 import BeautifulSoup
import os
import time

# Folder to save lyrics files
lyrics_folder = "data/lyrics"
os.makedirs(lyrics_folder, exist_ok=True)

# List of Zayn Malik song URLs (Add more if needed)
zayn_songs = [
    "https://genius.com/Zayn-pillowtalk-lyrics",
    "https://genius.com/Zayn-dusk-till-dawn-lyrics",
    "https://genius.com/Zayn-vibez-lyrics",
    "https://genius.com/Zayn-better-lyrics",
    
    "https://genius.com/Zayn-sour-diesel-lyrics",
    
    "https://genius.com/Zayn-tonight-lyrics",
    "https://genius.com/Zayn-she-lyrics",
    "https://genius.com/Zayn-entertainer-lyrics",
   
    "https://genius.com/Zayn-tio-lyrics",
    "https://genius.com/Zayn-what-i-am-lyrics",
   
    "https://genius.com/Zayn-befoUr-lyrics",
    "https://genius.com/Zayn-tightrope-lyrics",
    "https://genius.com/Zayn-unfuckwitable-lyrics"
]


def scrape_lyrics(song_url):
    response = requests.get(song_url)
    if response.status_code != 200:
        print(f"Failed to fetch {song_url}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')

    # New scraping approach - try all potential containers
    lyrics = ""

    # Method 1: Check for modern div-based lyrics containers (used in latest Genius pages)
    for div in soup.find_all("div", {"data-lyrics-container": "true"}):
        lyrics += div.get_text(separator="\n")

    # Method 2: Fallback to older page format (rare but exists)
    if not lyrics.strip():
        legacy_lyrics_container = soup.find("div", class_="lyrics")
        if legacy_lyrics_container:
            lyrics = legacy_lyrics_container.get_text(separator="\n")

    # Method 3: Final fallback - check for any <p> tags under div[data-lyrics-container]
    if not lyrics.strip():
        for paragraph in soup.select('div[data-lyrics-container="true"] p'):
            lyrics += paragraph.get_text(separator="\n")

    if not lyrics.strip():
        print(f"No lyrics found on {song_url}")
        return None

    return lyrics.strip()

def save_lyrics(song_title, lyrics):
    file_path = os.path.join(lyrics_folder, f"{song_title}.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(lyrics)

def scrape_all_songs():
    for url in zayn_songs:
        song_title = url.split("/")[-1].replace("-lyrics", "").replace("-", " ").title()
        print(f"Scraping lyrics for: {song_title}")

        lyrics = scrape_lyrics(url)

        if lyrics:
            save_lyrics(song_title, lyrics)
            print(f"Saved: {song_title}\n")
        else:
            print(f"Could not fetch lyrics for: {song_title}\n")

        time.sleep(2)  # Gentle delay to avoid being blocked

if __name__ == "__main__":
    scrape_all_songs()
