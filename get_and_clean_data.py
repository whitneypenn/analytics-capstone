import pandas as pd
import spotipy
import os

sp = spotipy.Spotify()
from spotipy.oauth2 import SpotifyClientCredentials
client_credentials_manager = SpotifyClientCredentials(client_id=os.getenv('SPOTIFY'), client_secret=os.getenv('SPOTIFY_SECRET_KEY'))
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace=False

def get_playlist_tracks(username,playlist_id):
    results = sp.user_playlist_tracks(username,playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

def get_popularity_and_ids(playlist):
    popularity = []
    ids = []
    for i in range(len(playlist)):
        if playlist[i]['track']['id'] is not None:
            popularity.append(playlist[i]['track']['popularity'])
            ids.append(playlist[i]['track']['id'])
    return popularity, ids

def get_attributes(id_list):
    results = []
    attributes_by_song = []
    slices = [id_list[i:i+50] for i in range(0, len(id_list), 50)]
    for i in slices:
        results.append(sp.audio_features(i))
    for i in results:
        for j in i:
            attributes_by_song.append(j)
    return attributes_by_song

def attributes_to_data_frame(attribute_list):
    df = pd.DataFrame(attribute_list)
    return df

my_playlist = get_playlist_tracks("Pitchfork", "spotify:user:pitchforkmedia:playlist:31mWsJSygA2Vx1FyyhXFS4")
pop, my_ids = get_popularity_and_ids(my_playlist)
my_attributes = get_attributes(my_ids)
df = attributes_to_data_frame(my_attributes)
