# Predicting Popularity of Spotify Songs

### Question: Can I accurately predict a song's popularity rating on Spotify based on song attributes?

## Data Source
### Spotify API
Spotify makes metadata available through their [API](https://developer.spotify.com/). To interact with the API, I used a python library called [Spotipy](https://github.com/plamere/spotipy). You can see the code I used to pull down data [here](https://github.com/whitneypenn/analytics-capstone/blob/master/src/get_and_clean_data.py)

I used 10 playlists as my sample of songs. I went for playlists curated by journalists, Spotify, and individuals, since I figured that's who's doing most of the curating on Spotify anyways. The 2000s are heavily sampled, since popularity is weighted for music that was listened to recently.

-  [Pitchfork's Top 500 Tracks of the 2000s](https://open.spotify.com/user/pitchforkmedia/playlist/31mWsJSygA2Vx1FyyhXFS4). Spotify's [All Out 00s](https://open.spotify.com/user/spotify/playlist/37i9dQZF1DX4o1oenSJRJd), [Country Gold](https://open.spotify.com/user/spotify/playlist/37i9dQZF1DWYnwbYQ5HnZU), [Rap Caviar](https://open.spotify.com/user/spotify/playlist/37i9dQZF1DX0XUsuxWHRQd), and [The Cookout](https://open.spotify.com/user/spotify/playlist/37i9dQZF1DXab8DipvnuNU). [New York Times Magazine's 2018 Music Playlist](https://open.spotify.com/user/nytmag/playlist/1fIoLrK0POksamXuvzbTee). [Spotifnation's Ultimate Rock Hits of All Time](https://open.spotify.com/user/7jt4w8i9zjsn36sngapjop302/playlist/291drwQ10IlkH0hf1TJcFk). ["Caribou's The Longest Playlist](https://open.spotify.com/user/cariboutheband/playlist/4Dg0J0ICj9kKTGDyFu0Cv4). [Touchepurley's 2017
 Hits](https://open.spotify.com/user/touchepurley/playlist/5cJXS1TnQhldZyI4ObwR7l). [Leonard Partoza Balang's Popular Songs from 2013 to Present](https://open.spotify.com/user/leonardbalang/playlist/0M3Xy7HCXJwPUbQKdOGt51). [Whitney Meer's Ultimate Pop](https://open.spotify.com/user/whitneypenn/playlist/7rpjDLKSDl3eXxUD1rAuMI?si=ZnLP4Nk0Re2yCWJGuLgiow).

## EDA + Feature Engineering
Spotify's Audio Features Object contains information for various metadata fields, outlined in full [here](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/). Relevant descriptions pulled below.

|  Target   |   Description |
| - | |
| popularity |  The popularity of the track. The value will be between 0 and 100, with 100 being the most popular. Spotify calculates the popularity by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. |

|  Feature  |  Description |
|-
| acousticness |  A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. |    
| danceability |  Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. |      
| duration_ms | 	The duration of the track in milliseconds. |
| energy | Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.  |
| instrumentalness | Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. |
| key | The key the track is in.  |
| liveness | Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. |  
| loudness | The overall loudness of a track in decibels (dB). |
| mode | Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.|
| speechiness | Speechiness detects the presence of spoken words in a track. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. |  
| tempo | The overall estimated tempo of a track in beats per minute (BPM). |
| time_signature	| An estimated overall time signature of a track. |
| valence | A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive  | |

#### Distribution of Popularity:
![Popularity_Dist](notebooks/popularity_hist.png)

#### Attributes By Popularity
![Attribute By Popularity](notebooks/mean_attribute_values.png)

#### Popularity vs (Select) Track Attributes
Selection of the most interesting below, see all of them in the images folder.

![Popularity vs Danceability ](notebooks/danceability_scatter_plt.png)

![Popularity vs Energy](notebooks/energy_scatter_plt.png)

![Popularity vs Loudness](notebooks/loudness_scatter_plt.png)

![Popularity vs Valence](notebooks/valence_scatter_plt.png)

## Feature Engineering

## Modeling

## Results

## Future Work

## References
