import os
import time
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

argparser = ArgumentParser("preprocess_dataset")
argparser.add_argument("--file-path", type=str, default="data/spotify_dataset.csv")
argparser.add_argument("--min-song-frequency", type=int, default=50)
argparser.add_argument("--min-songs-per-user", type=int, default=50)
argparser.add_argument("--max-song-frequency", type=int, default=1000)
argparser.add_argument("--save-path", type=str, default="data")


def read_csv_file(path: str) -> pd.DataFrame:
    # CSV 파일 읽기 (잘못된 줄은 건너뜀)
    df_playlist = pd.read_csv(path, skipinitialspace=True, on_bad_lines="skip")  # 잘못된 줄을 건너뜀
    return df_playlist


def select_columns_and_remove_duplicates(df_playlist: pd.DataFrame) -> pd.DataFrame:
    # 필요한 컬럼만 선택하고 중복 제거 (user_id, trackname, artistname 조합의 중복 제거)
    df_playlist = df_playlist[["user_id", "trackname", "artistname"]].drop_duplicates().reset_index(drop=True)
    return df_playlist


def remove_missing_values(df_playlist: pd.DataFrame) -> pd.DataFrame:
    # **누락된 값이 있는 행 제거**
    df_playlist.dropna(subset=["trackname", "artistname"], inplace=True)
    return df_playlist


def filter_songs_by_frequency(df_playlist: pd.DataFrame, min_frequency: int, max_frequency: int) -> pd.DataFrame:
    # ------------------ 노래 빈도 기반 필터링 시작 ------------------

    # 각 노래의 등장 빈도 계산
    song_frequency = df_playlist["song_id"].value_counts()

    # 빈도 기준으로 노래 필터링 (예: 10회 이상 등장한 노래만 남기기)
    valid_songs = song_frequency[(song_frequency >= min_frequency) & (song_frequency <= max_frequency)].index

    # 필터링된 노래만 남기기
    df_playlist = df_playlist[df_playlist["song_id"].isin(valid_songs)].reset_index(drop=True)
    return df_playlist


def assign_song_index(df_playlist: pd.DataFrame) -> pd.DataFrame:
    # song_id에 새로운 song_index 할당 (0부터 시작하는 연속적인 인덱스)
    unique_songs = df_playlist["song_id"].drop_duplicates().reset_index(drop=True)
    song_id_to_index = pd.Series(data=unique_songs.index, index=unique_songs.values).to_dict()

    df_playlist["song_index"] = df_playlist["song_id"].map(song_id_to_index)
    return df_playlist


def calculate_user_song_count(self, df_playlist: pd.DataFrame) -> pd.DataFrame:
    # 각 유저가 들은 노래 수 계산
    user_song_count = df_playlist.groupby("user_id")["song_index"].nunique().reset_index()
    user_song_count.columns = ["user_id", "song_count"]
    return user_song_count


def calculate_user_song_id_count(df_playlist: pd.DataFrame) -> pd.DataFrame:
    # 각 유저가 들은 노래 수 계산
    user_song_id_count = df_playlist.groupby("user_id")["song_id"].nunique().reset_index()
    user_song_id_count.columns = ["user_id", "song_count"]
    return user_song_id_count


def filter_users_by_song_count(user_song_count: pd.DataFrame, song_count_threshold: int) -> pd.DataFrame:
    filtered_users = user_song_count[(user_song_count["song_count"] >= song_count_threshold)]["user_id"]
    return filtered_users


def assign_user_index(df_playlist: pd.DataFrame, filtered_users: pd.DataFrame) -> pd.DataFrame:
    # 필터링된 유저만 남기기
    df_playlist = df_playlist[df_playlist["user_id"].isin(filtered_users)].reset_index(drop=True)

    # user_id에 새로운 user_index 할당 (0부터 시작하는 연속적인 인덱스)
    unique_users = df_playlist["user_id"].drop_duplicates().reset_index(drop=True)
    user_id_to_index = pd.Series(data=unique_users.index, index=unique_users.values).to_dict()

    # user_index 컬럼 생성
    df_playlist["user_index"] = df_playlist["user_id"].map(user_id_to_index)
    return df_playlist


def select_user_song_columns(df_playlist: pd.DataFrame) -> pd.DataFrame:
    # 필요한 컬럼만 선택
    df_user_songs = df_playlist[["user_index", "song_index"]].drop_duplicates().reset_index(drop=True)
    return df_user_songs


def create_sparse_matrix(df_user_songs: pd.DataFrame) -> csr_matrix:
    # 희소 행렬 생성
    user_indices = df_user_songs["user_index"].values
    song_indices = df_user_songs["song_index"].values

    num_users = user_indices.max() + 1
    num_songs = song_indices.max() + 1

    data = np.ones(len(user_indices))

    sparse_matrix = csr_matrix((data, (user_indices, song_indices)), shape=(num_users, num_songs))
    return sparse_matrix


def save_sparse_matrix_and_dataframes(
    sparse_matrix: csr_matrix,
    df_playlist: pd.DataFrame,
    save_path: str,
    min_song_frequency: int,
    min_songs_per_user: int,
) -> pd.DataFrame:
    # 희소 행렬 저장
    mmwrite(f"{save_path}/playlist_song_matrix_{min_song_frequency}_{min_songs_per_user}.mtx", sparse_matrix)

    # song_index, trackname, artistname 데이터프레임 저장
    song_info = df_playlist[["song_index", "trackname", "artistname"]].drop_duplicates().reset_index(drop=True)
    song_info.to_csv(f"{save_path}/song_info_{min_song_frequency}_{min_songs_per_user}.csv", index=False)

    # user_index, user_id 데이터프레임 저장
    user_info = df_playlist[["user_index", "user_id"]].drop_duplicates().reset_index(drop=True)
    user_info.to_csv(f"{save_path}/user_info_{min_song_frequency}_{min_songs_per_user}.csv", index=False)

    print("모든 작업이 완료되었습니다.")

    return sparse_matrix, song_info, user_info


def print_sparse_matrix_info(sparse_matrix: csr_matrix):
    total_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]
    non_zero_elements = sparse_matrix.nnz
    density = non_zero_elements / total_elements

    # 행렬의 크기 확인
    print(f"Shape of playlist_song_matrix: {sparse_matrix.shape}")

    # 총 요소의 수
    print(f"총 요소의 수: {sparse_matrix.nnz}")

    print(f"희소행렬의 밀도: {density:.8f}")


def data_clean_up(df_playlist: pd.DataFrame) -> pd.DataFrame:

    df_playlist.columns = df_playlist.columns.str.replace('"', "")
    df_playlist.columns = df_playlist.columns.str.replace(" ", "")
    df_playlist.columns = df_playlist.columns.str.lower()

    return df_playlist


def data_load(file_path: str):
    df_playlist = read_csv_file(args.file_path)
    df_playlist = data_clean_up(df_playlist)

    # 결측치와 중복 제거
    df_playlist = select_columns_and_remove_duplicates(df_playlist)
    df_playlist = remove_missing_values(df_playlist)

    # 각 노래에 고유한 식별자(song_id) 생성 (trackname과 artistname의 조합)
    df_playlist["song_id"] = df_playlist[["trackname", "artistname"]].apply(lambda x: "_".join(x), axis=1)

    return df_playlist


def filter_users_by_song_count_top(
    user_song_id_count: pd.DataFrame, song_count_top_threshold: float = 0.99
) -> pd.DataFrame:

    threshold = user_song_id_count["song_count"].quantile(song_count_top_threshold)
    filtered_users = user_song_id_count[user_song_id_count["song_count"] <= threshold]["user_id"]
    return filtered_users


def main(args: Namespace):
    # 데이터 로드
    df_playlist = data_load(args.file_path)
    num_users = df_playlist["user_id"].nunique()
    num_songs = df_playlist["song_id"].nunique()

    updated_num_songs = 10e10
    updated_num_users = 10e10

    print(
        f"데이터 이상치 제거 기준: 유저당 노래 수 하한 -{args.min_songs_per_user}, 총 노래 등장 횟 수 하한 -{args.min_song_frequency}"
    )

    state = True
    while True:
        df_playlist = filter_songs_by_frequency(df_playlist, args.min_song_frequency, args.max_song_frequency)
        user_song_id_count = calculate_user_song_id_count(df_playlist)
        filtered_users = filter_users_by_song_count(user_song_id_count, args.min_songs_per_user)
        df_playlist = df_playlist[df_playlist["user_id"].isin(filtered_users.values)]

        if state:
            # 너무 큰 유저 제거
            filtered_users = filter_users_by_song_count_top(user_song_id_count)
            df_playlist = df_playlist[df_playlist["user_id"].isin(filtered_users.values)]

            state = False

        updated_num_users = df_playlist["user_id"].nunique()
        updated_num_songs = df_playlist["song_id"].nunique()
        total_rows = df_playlist.shape[0]

        # print(f"filtered_users: {updated_num_users}, filtered_songs: {updated_num_songs}, total_rows: {total_rows}")

        if updated_num_users == num_users and updated_num_songs == num_songs:
            break
        else:
            num_users = updated_num_users
            num_songs = updated_num_songs

    print(f"filtered_users: {updated_num_users}, filtered_songs: {updated_num_songs}, total_rows: {total_rows}")

    if updated_num_users * updated_num_songs == 0:
        return

    df_playlist = assign_user_index(df_playlist, filtered_users)
    df_playlist = assign_song_index(df_playlist)

    # 희소행령 생성
    df_user_songs = select_user_song_columns(df_playlist)
    sparse_matrix = create_sparse_matrix(df_user_songs)

    # 희소 행렬 저장
    save_sparse_matrix_and_dataframes(
        sparse_matrix, df_playlist, args.save_path, args.min_song_frequency, args.min_songs_per_user
    )
    print_sparse_matrix_info(sparse_matrix)

    return


if __name__ == "__main__":
    args = argparser.parse_args()

    main(args)
