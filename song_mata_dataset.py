import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmwrite

from preprocess_dataset import DataPreprocess
import requests
import base64
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import time
from unidecode import unidecode
from fuzzywuzzy import fuzz

from spotipy.oauth2 import SpotifyClientCredentials

def spotify_api_init():
    # Replace with your own Client ID and Client Secret
    CLIENT_ID = 'ae73429356644539bd05e79773e6a069'
    CLIENT_SECRET = 'f65bd31e1a664a2ab8e71ed3232119a5'

    # Client Credentials Flow 사용
    client_credentials_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
    
    sp = spotipy.Spotify(auth_manager=client_credentials_manager)
    return sp

def normalize_string(s):
    return unidecode(s.strip()).lower()

def get_best_match(results, track_name, artist_name):
    highest_score = 0
    best_match = None
    for item in results['tracks']['items']:
        result_track = item['name']
        result_artist = ', '.join([artist['name'] for artist in item['artists']])
        score = fuzz.token_set_ratio(f"{result_track} {result_artist}", f"{track_name} {artist_name}")
        if score > highest_score:
            highest_score = score
            best_match = item
    return best_match if highest_score > 80 else None  # Adjust the threshold as needed

def collect_songs_data(input_df, sp):
    all_songs_data = pd.DataFrame()
    songs_list = []

    for idx, row in input_df.iterrows():
        if idx > 100:
            break

        track_name = normalize_string(row['trackname'])
        artist_name = normalize_string(row['artistname'])
        
        query = f'track_name:"{track_name}" artist_name:"{artist_name}"'
        
        try:
            results = sp.search(q=query, type='track', limit=5, market='US')
            if results['tracks']['items']:
                best_match = get_best_match(results, track_name, artist_name)
                if best_match:
                    track_id = best_match['id']
                    # 트랙의 전체 정보 가져오기
                    track_info = sp.track(track_id)
                    
                    # 필요한 정보 추출
                    song_data = {
                        'song_index': row['song_index'],
                        'track_id': track_id,
                        'track_name': track_info['name'],
                        'artist_name': track_info['artists'][0]['name'],
                        'album_name': track_info['album']['name'],
                        'popularity': track_info['popularity'],
                        'duration_ms': track_info['duration_ms'],
                        'explicit': track_info['explicit'],
                        'preview_url': track_info.get('preview_url'),
                        'external_url': track_info['external_urls']['spotify']
                    }
                    songs_list.append(song_data)
                    print(f"✅ Got info for track: {track_name} by {artist_name}")
                else:
                    print(f"❌ No close match found for: {track_name} by {artist_name}")
            else:
                print(f"❌ No results found for: {track_name} by {artist_name}")
        except Exception as e:
            print(f"Error processing track {track_name}: {str(e)}")
            continue

    if songs_list:
        all_songs_data = pd.DataFrame(songs_list)
        print(f"Successfully collected data for {len(songs_list)} tracks")
        print(all_songs_data.shape)
        return all_songs_data
    else:
        print("No track data collected")
        return pd.DataFrame()

def save_df(df: pd.DataFrame, save_path: str):
    df.to_csv(save_path, index=False)

def main(data_preprocess: DataPreprocess, save_path: str):
    df_playlist = data_preprocess.read_csv_file("data/spotify_dataset.csv")
    df_playlist = data_preprocess.data_clean_up(df_playlist)
    print(df_playlist.shape)
    df_playlist = data_preprocess.select_columns_and_remove_duplicates(df_playlist)
    df_playlist = data_preprocess.remove_missing_values(df_playlist)



    for i in range(0, 10):
        df_playlist = data_preprocess.filter_songs_by_frequency(df_playlist, 30)
        user_song_id_count = data_preprocess.calculate_user_song_id_count(df_playlist)
        filtered_users = data_preprocess.filter_users_by_song_count(user_song_id_count, 0.005, 3)
        df_playlist = df_playlist[df_playlist['user_id'].isin(filtered_users.values)]
        print(f"i: {i}, filtered_users.shape: {filtered_users.shape}, df_playlist.shape: {df_playlist.shape}")


    df_playlist = data_preprocess.assign_user_index(df_playlist, filtered_users)
    df_playlist = data_preprocess.assign_song_index(df_playlist)
    df_user_songs = data_preprocess.select_user_song_columns(df_playlist)
    sparse_matrix = data_preprocess.create_sparse_matrix(df_user_songs)
    data_preprocess.save_sparse_matrix_and_dataframes(sparse_matrix, df_playlist, 30, save_path)
    data_preprocess.print_sparse_matrix_info(sparse_matrix)
    

    return

def main2(data_preprocess: DataPreprocess, save_path: str, df_playlist: pd.DataFrame):
    sp = spotify_api_init()

    # 곡 데이터 수집
    all_songs_data = collect_songs_data(df_playlist, sp)
    
    # 결과 저장
    save_df(all_songs_data, save_path + "song_data.csv")

if __name__ == "__main__":
    data_preprocess = DataPreprocess()
    save_path = "data2/"

    if os.path.exists(save_path + "song_info_30.csv"):
        df_song_info = data_preprocess.read_csv_file(save_path + "song_info_30.csv")
    else:
        main(data_preprocess, save_path)
        df_song_info = data_preprocess.read_csv_file(save_path + "song_info_30.csv")

    main2(data_preprocess, save_path, df_song_info)
