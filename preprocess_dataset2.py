
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmwrite



def data_load():
    df_playlist = pd.read_csv("data/spotify_dataset.csv")
    return df_playlist

def data_clean_up(df_playlist):

    df_playlist.columns = df_playlist.columns.str.replace('"', '')
    df_playlist.columns = df_playlist.columns.str.replace('name', '')
    df_playlist.columns = df_playlist.columns.str.replace(' ', '')
    df_playlist.columns = df_playlist.columns.str.lower()

    return df_playlist


def read_csv_file(path: str):
    # CSV 파일 읽기 (잘못된 줄은 건너뜀)
    df_playlist = pd.read_csv(
        path,
        skipinitialspace=True,
        on_bad_lines='skip'  # 잘못된 줄을 건너뜀
    )
    return df_playlist

def select_columns_and_remove_duplicates(df_playlist: pd.DataFrame) -> pd.DataFrame:
    # 필요한 컬럼만 선택하고 중복 제거 (user_id, trackname, artistname 조합의 중복 제거)
    df_playlist = df_playlist[['user_id', 'trackname', 'artistname']].drop_duplicates().reset_index(drop=True)
    return df_playlist

def remove_missing_values(df_playlist):
    # **누락된 값이 있는 행 제거**
    df_playlist.dropna(subset=['trackname', 'artistname'], inplace=True)
    return df_playlist

def filter_songs_by_frequency(df_playlist, min_frequency):
    # ------------------ 노래 빈도 기반 필터링 시작 ------------------

    # 각 노래에 고유한 식별자 생성 (trackname과 artistname의 조합)
    df_playlist['song_id'] = df_playlist[['trackname', 'artistname']].apply(lambda x: '_'.join(x), axis=1)

    # 각 노래의 등장 빈도 계산
    song_frequency = df_playlist['song_id'].value_counts()

    # 빈도 기준으로 노래 필터링 (예: 10회 이상 등장한 노래만 남기기)
    valid_songs = song_frequency[song_frequency >= min_frequency].index

    # 필터링된 노래만 남기기
    df_playlist = df_playlist[df_playlist['song_id'].isin(valid_songs)].reset_index(drop=True)
    return df_playlist

def assign_song_index(df_playlist):
    # song_id에 새로운 song_index 할당 (0부터 시작하는 연속적인 인덱스)
    unique_songs = df_playlist['song_id'].drop_duplicates().reset_index(drop=True)
    song_id_to_index = pd.Series(data=unique_songs.index, index=unique_songs.values).to_dict()

    df_playlist['song_index'] = df_playlist['song_id'].map(song_id_to_index)
    return df_playlist

def calculate_user_song_count(df_playlist):
    # 각 유저가 들은 노래 수 계산
    user_song_count = df_playlist.groupby('user_id')['song_index'].nunique().reset_index()
    user_song_count.columns = ['user_id', 'song_count']
    return user_song_count

def filter_users_by_song_count(user_song_count, lower_percent, mean_percent):
    # 평균 및 표준편차 계산
    mean_count = user_song_count['song_count'].mean()
    std_count = user_song_count['song_count'].std()

    lower_percentile = user_song_count['song_count'].quantile(lower_percent)

    # 평균 ± 2*표준편차 이내의 범위로 필터링 and 하위 20% 버리기
    filtered_users = user_song_count[
        (user_song_count['song_count'] >= mean_count - mean_percent * std_count) &
        (user_song_count['song_count'] <= mean_count + mean_percent * std_count) &
        (user_song_count['song_count'] >= lower_percentile)
    ]['user_id']
    return filtered_users

def assign_user_index(df_playlist, filtered_users):
    # 필터링된 유저만 남기기
    df_playlist = df_playlist[df_playlist['user_id'].isin(filtered_users)].reset_index(drop=True)

    # user_id에 새로운 user_index 할당 (0부터 시작하는 연속적인 인덱스)
    unique_users = df_playlist['user_id'].drop_duplicates().reset_index(drop=True)
    user_id_to_index = pd.Series(data=unique_users.index, index=unique_users.values).to_dict()

    # user_index 컬럼 생성
    df_playlist['user_index'] = df_playlist['user_id'].map(user_id_to_index)
    return df_playlist

def select_user_song_columns(df_playlist):
    # 필요한 컬럼만 선택
    df_user_songs = df_playlist[['user_index', 'song_index']].drop_duplicates().reset_index(drop=True)
    return df_user_songs

def create_sparse_matrix(df_user_songs):
    # 희소 행렬 생성
    user_indices = df_user_songs['user_index'].values
    song_indices = df_user_songs['song_index'].values

    num_users = user_indices.max() + 1
    num_songs = song_indices.max() + 1

    data = np.ones(len(user_indices))

    sparse_matrix = csr_matrix((data, (user_indices, song_indices)), shape=(num_users, num_songs))
    return sparse_matrix


def save_sparse_matrix_and_dataframes(sparse_matrix, df_playlist, min_frequency, save_path):
    # 희소 행렬 저장
    mmwrite(f'{save_path}/playlist_song_matrix_{min_frequency}.mtx', sparse_matrix)

    # song_index, trackname, artistname 데이터프레임 저장
    song_info = df_playlist[['song_index', 'trackname', 'artistname']].drop_duplicates().reset_index(drop=True)
    song_info.to_csv(f'{save_path}/song_info_{min_frequency}.csv', index=False)

    # user_index, user_id 데이터프레임 저장
    user_info = df_playlist[['user_index', 'user_id']].drop_duplicates().reset_index(drop=True)
    user_info.to_csv(f'{save_path}/user_info_{min_frequency}.csv', index=False)

    print("모든 작업이 완료되었습니다.")


def print_sparse_matrix_info(sparse_matrix):
    total_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]
    non_zero_elements = sparse_matrix.nnz
    density = non_zero_elements / total_elements

    # 행렬의 크기 확인
    print(f"Shape of playlist_song_matrix: {sparse_matrix.shape}")

    # 총 요소의 수
    print(f"총 요소의 수: {sparse_matrix.nnz}")

    print(f"희소행렬의 밀도: {density:.8f}")




def main():

    file_path = f"/home/comoz/main_project/playlist_project/data/spotify_dataset.csv"
    min_frequency = 50
    lower_percent = 0.1
    mean_percent = 10
    save_path = "/home/comoz/main_project/playlist_project/data/dataset"

    df_playlist = read_csv_file(file_path)

    df_playlist = select_columns_and_remove_duplicates(df_playlist)
    df_playlist = remove_missing_values(df_playlist)


    df_playlist = filter_songs_by_frequency(df_playlist, min_frequency)
    df_playlist = assign_song_index(df_playlist)

    
    user_song_count = calculate_user_song_count(df_playlist)
    filtered_users = filter_users_by_song_count(user_song_count, lower_percent, mean_percent)
    df_playlist = assign_user_index(df_playlist, filtered_users)
    df_user_songs = select_user_song_columns(df_playlist)
    sparse_matrix = create_sparse_matrix(df_user_songs)
    save_sparse_matrix_and_dataframes(sparse_matrix, df_playlist, min_frequency, save_path)
    print_sparse_matrix_info(sparse_matrix)

if __name__ == "__main__":
    main()