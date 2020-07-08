import pandas as pd
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def get_track_state(x):
    if x >= 20:
        state = 'P'  # poor: 불량
    elif 15 <= x < 20:
        state = 'S'  # saturated: 포화
    elif 10 <= x < 15:
        state = 'H'  # humid: 다습
    elif 6 <= x < 10:
        state = 'G'  # good: 양호
    elif 1 <= x < 5:
        state = 'D'  # dry: 건조
    else:
        state = 'N'  # 해당없음

    return state


def race_result_features(df):
    df['track_state_new'] = df['track_state'].apply(lambda x: get_track_state(x))
    dummy_df = pd.get_dummies(df['track_state_new'], prefix='track_state')
    dummy_df.drop(columns=['track_state_P'], axis=1, inplace=True)

    all_entire_dummy_columns = ['track_state_P', 'track_state_S', 'track_state_H', 'track_state_G', 'track_state_D', 'track_state_N']
    if len(all_entire_dummy_columns) != len(list(dummy_df.columns)):
        for i in list(set(all_entire_dummy_columns)-set(list(dummy_df.columns))) :
            dummy_df[i] = 0

    df = pd.concat([df, dummy_df], axis=1)
    df.drop(['track_state', 'track_state_new'], axis=1, inplace=True)
    df.rename(columns=lambda x: 'race_' + x if x == 'track_length' or 'track_state' in x or x == 'weight' else x, inplace=True)

    return df


def horse_features(df):
    # ['한' '미' '뉴' '한(포)' '캐' '일' '호' '남' '아일' '모' '영' '프' '브']
    df['home'].replace({'한': 'domestic', '한(포)': 'domestic', '미': 'foreign', '뉴': 'foreign', '캐': 'foreign',
                        '일': 'foreign', '호': 'foreign', '남': 'foreign', '아일': 'foreign', '모': 'foreign',
                        '영': 'foreign', '프': 'foreign', '브': 'foreign'}, inplace=True)

    # ['수' '암' '거']
    df['gender'].replace({'수': 'M', '암': 'F', '거': 'N'}, inplace=True)

    all_home_dummy_columns = ['horse_home_domestic', 'horse_home_foreign']
    home_dummy_df = pd.get_dummies(df['home'], prefix='home')
    if len(all_home_dummy_columns) != len(list(home_dummy_df.columns)):
        for i in list(set(all_home_dummy_columns)-set(list(home_dummy_df.columns))) :
            home_dummy_df[i] = 0
        # add_df = pd.DataFrame(0, columns=list(set(all_home_dummy_columns)-set(list(home_dummy_df.columns))))
        # home_dummy_df = pd.concat([home_dummy_df, add_df], axis=1)

    all_gender_dummy_columns = ['horse_gender_F', 'horse_gender_M', 'horse_gender_N']
    gender_dummy_df = pd.get_dummies(df['gender'], prefix='gender')
    if len(all_gender_dummy_columns) != len(list(gender_dummy_df.columns)):
        for i in list(set(all_gender_dummy_columns)-set(list(gender_dummy_df.columns))) :
            gender_dummy_df[i] = 0
        # add_df = pd.DataFrame(0, columns=list(set(all_gender_dummy_columns)-set(list(gender_dummy_df.columns))))
        # gender_dummy_df = pd.concat([gender_dummy_df, add_df], axis=1)

    df = pd.concat([df, home_dummy_df, gender_dummy_df], axis=1)

    df['class'] = df['class'].apply(lambda x: 6 if len(list(x)) < 2 else (6 if list(x)[1] == '미' else int(list(x)[1])))
    df['total_win_ratio'] = (df['first'] + df['second']) / df['race_count']
    df['total_win_ratio'] = df['total_win_ratio'].fillna(0)
    df['1yr_win_ratio'] = (df['1yr_first'] + df['1yr_second']) / df['1yr_count']
    df['1yr_win_ratio'] = df['1yr_win_ratio'].fillna(0)

    df.drop(['home', 'gender', 'first', 'second', 'race_count', '1yr_first', '1yr_second', '1yr_count'], axis=1,
            inplace=True)
    df.rename(columns=lambda x: 'horse_' + x if x != 'date' and x != 'horse' else x, inplace=True)

    return df


def jockey_features(df):

    df['date_year'] = df['date'].apply(lambda x: x.year)
    df['debut'] = df['debut'].astype('datetime64[ns]')
    df['debut_year'] = df['debut'].apply(lambda x: x.year)
    df['career'] = df['date_year'] - df['debut_year']

    df['total_win_ratio'] = (df['first'] + df['second']) / df['race_count']
    df['total_win_ratio'] = df['total_win_ratio'].fillna(0)
    df['1yr_win_ratio'] = (df['1yr_first'] + df['1yr_second']) / df['1yr_count']
    df['1yr_win_ratio'] = df['1yr_win_ratio'].fillna(0)

    df.drop(['date_year', 'debut_year', 'debut', 'first', 'second', 'race_count',
             '1yr_first', '1yr_second', '1yr_count'], axis=1, inplace=True)
    df.rename(columns=lambda x: 'jockey_' + x if x != 'date' and x != 'jockey' else x, inplace=True)

    return df


def owner_features(df):
    # '1yr_count', '1yr_first', '1yr_second', '1yr_third', 'race_count', 'first', 'second', 'third'
    df['total_win_ratio'] = (df['first'] + df['second'] + df['third']) / df['race_count']
    df['total_win_ratio'] = df['total_win_ratio'].fillna(0)
    df['1yr_win_ratio'] = (df['1yr_first'] + df['1yr_second'] + df['1yr_third']) / df['1yr_count']
    df['1yr_win_ratio'] = df['1yr_win_ratio'].fillna(0)

    df.drop(['1yr_count', '1yr_first', '1yr_second', '1yr_third',
             'race_count', 'first', 'second', 'third'], axis=1, inplace=True)
    df.rename(columns=lambda x: 'owner_' + x if x != 'date' and x != 'owner' else x, inplace=True)

    return df


def trainer_features(df):
    # remove strange raw data
    df = df[df['age'] > 0]

    df['date_year'] = df['date'].apply(lambda x: x.year)
    df['debut'] = df['debut'].astype('datetime64[ns]')
    df['debut_year'] = df['debut'].apply(lambda x: x.year)
    df['career'] = df['date_year'] - df['debut_year']

    df['total_win_ratio'] = (df['first'] + df['second']) / df['race_count']
    df['total_win_ratio'] = df['total_win_ratio'].fillna(0)
    df['1yr_win_ratio'] = (df['1yr_first'] + df['1yr_second']) / df['1yr_count']
    df['1yr_win_ratio'] = df['1yr_win_ratio'].fillna(0)

    df.drop(['date_year', 'debut_year', 'debut', 'first', 'second', 'race_count',
             '1yr_first', '1yr_second', '1yr_count'], axis=1, inplace=True)
    df.rename(columns=lambda x: 'trainer_' + x if x != 'date' and x != 'trainer' else x, inplace=True)

    return df



