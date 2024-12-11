import base64
import os
import time
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import requests
import spotipy
from fuzzywuzzy import fuzz
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from unidecode import unidecode

argparser = ArgumentParser("api_data_collect")
argparser.add_argument("--file-path", type=str, default="data/spotify_dataset.csv")
argparser.add_argument("--save-path", type=str, default="data")
argparser.add_argument("--client-id", type=str, default="ae73429356644539bd05e79773e6a069")
argparser.add_argument("--client-secret", type=str, default="f65bd31e1a664a2ab8e71ed3232119a5")
argparser.add_argument("--redirect-uri", type=str, default="http://localhost:8080")
argparser.add_argument(
    "--scope",
    type=str,
    default="playlist-read-private user-library-read user-read-private playlist-modify-public playlist-modify-private",
)


def spotify_api_init(client_id: str, client_secret: str, redirect_uri: str, scope: str, access_token: str):

    # OAuth 2.0 인증 사용
    # auth_manager = spotipy.oauth2.SpotifyOAuth(
    #     client_id=client_id,
    #     client_secret=client_secret,
    #     redirect_uri=redirect_uri,
    #     scope=scope,
    # )

    sp = spotipy.Spotify(auth=access_token)
    return sp


def normalize_string(s):
    return unidecode(s.strip()).lower()


def get_best_match(results, track_name, artist_name):
    highest_score = 0
    best_match = None
    for item in results["tracks"]["items"]:
        result_track = item["name"]
        result_artist = ", ".join([artist["name"] for artist in item["artists"]])
        score = fuzz.token_set_ratio(f"{result_track} {result_artist}", f"{track_name} {artist_name}")
        if score > highest_score:
            highest_score = score
            best_match = item
    return best_match if highest_score > 80 else None  # Adjust the threshold as needed


def collect_songs_data(input_df, sp):
    all_songs_data = pd.DataFrame()
    songs_list = []
    track_id_list = []

    for idx, row in input_df.iterrows():
        if idx > 100:
            break

        track_name = normalize_string(row["trackname"])
        artist_name = normalize_string(row["artistname"])

        query = f'track_name:"{track_name}" artist_name:"{artist_name}"'

        results = sp.search(q=query, type="track", limit=5, market="US")
        if results["tracks"]["items"]:
            best_match = get_best_match(results, track_name, artist_name)
            if best_match:
                track_id = best_match.get("id")
                print(f"track_id: {track_id}")
                print(f"Found track: {track_name} by {artist_name}")
                if track_id:  # None이 아닌 경우에만 추가
                    track_id_list.append(track_id)
            else:
                print(f"❌ No close match found for: {track_name} by {artist_name}")
                continue

    print(f"Total tracks collected: {len(track_id_list)}")
    print(f"First few track IDs: {track_id_list[:5]}")

    # 모든 audio features를 저장할 리스트
    all_audio_features = []

    # 각 트랙에 대해 개별적으로 audio_features 요청
    for track_id in track_id_list:
        try:
            # 개별 트랙에 대한 audio_features 요청
            audio_feature = sp.audio_features(track_id)[0]
            if audio_feature:
                all_audio_features.append(audio_feature)
                print(f"✅ Got features for track: {track_id}")
            time.sleep(0.1)  # API 호출 사이에 잠시 대기
        except Exception as e:
            print(f"Error getting audio features for track {track_id}: {str(e)}")
            continue

    if all_audio_features:
        features_df = pd.DataFrame(all_audio_features)
        print(f"Successfully retrieved audio features for {len(all_audio_features)} tracks")
        print(features_df.shape)
        return features_df
    else:
        print("No audio features collected")
        return pd.DataFrame()


def save_df(df: pd.DataFrame, save_path: str):
    df.to_csv(save_path, index=False)


def main(args: Namespace, access_token: str):
    sp = spotify_api_init(args.client_id, args.client_secret, args.redirect_uri, args.scope, access_token)

    # CSV 파일 읽기 (잘못된 줄은 건너뜀)
    df_playlist = pd.read_csv(args.file_path, skipinitialspace=True, on_bad_lines="skip")  # 잘못된 줄을 건너뜀

    # 곡 데이터 수집
    all_songs_data = collect_songs_data(df_playlist, sp)

    # 결과 저장
    save_df(all_songs_data, save_path + "song_data.csv")


def test(args: Namespace):
    import base64

    import requests

    # Replace with your own Client ID and Client Secret
    CLIENT_ID = args.client_id
    CLIENT_SECRET = args.client_secret

    # Base64 encode the client ID and client secret
    client_credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    client_credentials_base64 = base64.b64encode(client_credentials.encode())

    # Request the access token
    token_url = "https://accounts.spotify.com/api/token"
    headers = {"Authorization": f"Basic {client_credentials_base64.decode()}"}
    data = {
        "grant_type": "client_credentials",
    }
    response = requests.post(token_url, data=data, headers=headers)

    if response.status_code == 200:
        access_token = response.json()["access_token"]
        print("Access token obtained successfully.")
        print(access_token)

        return access_token
    else:
        print("Error obtaining access token.")
        exit()


if __name__ == "__main__":
    args = argparser.parse_args()
    save_path = "data/"
    # main(args)
    access_token = test(args)
    main(args, access_token)
