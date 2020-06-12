import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from .technical_indicator import *


def get_data_path(symbol):
    # Return CSV file path given symbol.
    commodity_dir = '../data/commodities'
    currency_dir = '../data/currencies'

    if symbol in ['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD']:
        path = os.path.join(currency_dir, symbol + '.csv')
    else:
        path = os.path.join(commodity_dir, symbol + '.csv')

    return path


def merge_data(start_date, end_date, symbols):
    dates = pd.date_range(start_date, end_date, freq='D')
    df = pd.DataFrame(index=dates)

    if 'USD' not in symbols:
        symbols.insert(0, 'USD')

    for symbol in symbols:
        df_temp = pd.read_csv(get_data_path(symbol), index_col="Date", parse_dates=True, na_values=['nan'])
        df_temp = df_temp.reindex(dates)
        df_temp.columns = [symbol + '_' + col for col in df_temp.columns]  # rename columns
        df = df.join(df_temp)

    return df


def min_max_scaler(data):
    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min
    norm_data = (data-data.min()) / (data.max() - data.min())
    return norm_data


def get_selected_columns(clustering_result):
    random.seed(123)
    cluster_dict = dict()
    for idx, n_cluster in enumerate(clustering_result.labels_):
        if n_cluster in cluster_dict.keys():
            cluster_dict[n_cluster].append(idx)
        else:
            cluster_dict[n_cluster] = [idx]

    reprsentative_columns_list = []
    # get representative column for each cluster, select one col index from list randomly
    for key in cluster_dict.keys():
        reprsentative_columns_list.append(random.choice(cluster_dict[key]))

    return reprsentative_columns_list


def select_features(features_df, USD_price, n_clusters):
    input_days = 10

    # k-means clustering
    if n_clusters > features_df.shape[1]:
        print("Set number of clusters to maximum size of training set")
        n_clusters = features_df.shape[1]
    clustered_results = KMeans(n_clusters=n_clusters).fit(features_df.transpose())
    selected_columns = get_selected_columns(clustered_results)
    selected_features_df = features_df.iloc[:, selected_columns]

    x = windowing_x(selected_features_df, input_days)
    y = windowing_y(USD_price, input_days)

    # split training and test data
    training_x = x[:-10]
    training_y = y[:-10]

    test_x = x[-10]
    test_y = y[-10]

    return list(selected_features_df.columns), training_x, training_y, test_x, test_y


# def make_raw_features_original(start_date, end_date):
#     # Select symbols
#     # currency : AUD, CNY, EUR, GBP, HKD, JPY, USD
#
#     if os.path.exists('features/raw_data.pkl'):
#         all_features_df = pd.read_pickle('features/raw_data.pkl')
#     else:
#         # Remove high correlation currency HKD
#         symbols_list = ['AUD', 'CNY', 'EUR', 'GBP', 'JPY', 'USD', 'BrentOil', 'Copper', 'CrudeOil', 'Gasoline', 'Gold', 'NaturalGas', 'Platinum', 'Silver']
#         raw_table = merge_data(start_date, end_date, symbols=symbols_list)
#         # raw_table.dropna(inplace=True, how='all')
#         raw_table.dropna(inplace=True)
#         raw_table.fillna(method='ffill', inplace=True).dropna()
#
#         # TODO
#         # 1. Open, High, Low column 모두 없애기
#         # 2. technical indicators 만든 dataframe에서 USD_Price와 USD_Change와 correlation 높은 것들 제외
#         all_features_df = pd.DataFrame()
#
#         # Use currency features
#         for currency in symbols_list:
#             curr_currency_columns = [currency+"_Open", currency+"_High", currency+"_Low", currency+"_Price", currency+"_Change"]
#             curr_df = raw_table[curr_currency_columns]
#
#             curr_df = ta.add_trend_ta(curr_df, currency+"_High", currency+"_Low", currency+"_Price", fillna=True, colprefix=currency+"_")
#             curr_df = ta.add_volatility_ta(curr_df, currency+"_High", currency+"_Low", currency+"_Price", fillna=True, colprefix=currency+"_")
#             curr_df = ta.add_others_ta(curr_df, currency+"_Price", fillna=True, colprefix=currency+"_")
#             curr_df = ta.add_momentum_ta(curr_df, currency+"_High", currency+"_Low", currency+"_Price", currency+"_Vol",fillna=True, colprefix=currency+"_")
#
#             # Remove Open, High, Low columns since we are predicting daily close price
#             curr_df = curr_df[list(set(curr_df.columns) - set([currency + '_Open', currency + '_High', currency + '_Low']))]
#
#             all_features_df = pd.concat([all_features_df, curr_df], axis=1)
#             print(currency, all_features_df.shape, curr_df.shape)
#
#         all_features_df.to_pickle('features/raw_data.pkl')
#
#     # all_features_df.dropna(inplace=True, axis='columns', how='all')
#     # all_features_df = all_features_df.fillna(method='bfill').fillna(method='ffill')
#     print('final: {}'.format(all_features_df.shape))
#
#     max_num_columns = all_features_df.shape[1]
#
#     USD_price = all_features_df['USD_Price']
#     # TODO remove USD_price? why?
#     features_list = list(all_features_df.columns)
#     features_list.remove('USD_Price')
#
#     # Since MLPRegressor is very sensitive to numeric scale, do min_max_scaling
#     features_df = all_features_df[features_list]
#     features_df = min_max_scaler(features_df)
#
#     return features_df, USD_price, max_num_columns


def make_raw_features(start_date, end_date, reuse=False):
    if reuse and os.path.exists('features/raw_data.pkl'):
        all_features_df = pd.read_pickle('features/raw_data.pkl')
    else:
        currency_symbols = ['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY']
        commodity_symbols = ['BrentOil', 'Copper', 'CrudeOil', 'Gasoline', 'Gold', 'NaturalGas', 'Platinum', 'Silver']

        currency_df = merge_data(start_date, end_date, symbols=currency_symbols)
        currency_df = currency_df[[col for col in currency_df.columns if 'Change' not in col]]
        currency_df.dropna(inplace=True)

        commodity_df = merge_data(start_date, end_date, symbols=commodity_symbols)
        commodity_df = commodity_df[[col for col in commodity_df.columns if 'Volume' not in col]]
        commodity_df = commodity_df[[col for col in commodity_df.columns if 'Change' not in col]]
        commodity_df.dropna(inplace=True)

        total_table = pd.concat([currency_df, commodity_df.iloc[:, 4:]], axis=1)
        total_table = total_table.fillna(method='ffill').dropna()

        for sym in currency_symbols + commodity_symbols:
            add_trend_ta(total_table, sym + '_High', sym + '_Low', sym + '_Price', fillna=True, colprefix=sym + '_')
            add_volatility_ta(total_table, sym + '_High', sym + '_Low', sym + '_Price', fillna=True, colprefix=sym + '_')
            add_momentum_ta(total_table, sym+"_High", sym+"_Low", sym+"_Price", sym+"_Vol",fillna=True, colprefix=sym+"_")
            add_others_ta(total_table, sym + '_Price', fillna=True, colprefix=sym + '_')

        all_features_df = total_table
        all_features_df.to_pickle('features/raw_data.pkl')
    # print('preprocessed features: {}'.format(all_features_df.shape))

    max_num_columns = all_features_df.shape[1]

    USD_price = all_features_df['USD_Price']
    # TODO remove USD_price? why?
    features_list = list(all_features_df.columns)
    features_list.remove('USD_Price')

    # Since MLPRegressor is very sensitive to numeric scale, do min_max_scaling
    features_df = all_features_df[features_list]
    features_df = min_max_scaler(features_df)

    return features_df, USD_price, max_num_columns


def windowing_y(data, input_days):
    windows = [data[i + input_days:i + input_days + 10] for i in range(len(data) - input_days)]
    return windows


def windowing_x(data, input_days):
    # flattening into 1-D array
    windows = np.array([data[i:i + input_days].values.ravel() for i in range(len(data) - input_days)])

    return windows


if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2020-04-06'

    make_raw_features(start_date, end_date)
