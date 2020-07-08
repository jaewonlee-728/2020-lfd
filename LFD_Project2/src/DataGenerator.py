import os
import pandas as pd
import numpy as np
import random
from technical_indicator import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans


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
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    if 'Gold' not in symbols:
        symbols.insert(0, 'Gold')

    for symbol in symbols:
        df_temp = pd.read_csv(get_data_path(symbol), index_col="Date", parse_dates=True, na_values=['nan'])
        df_temp.columns = [symbol + '_' + col for col in df_temp.columns]  # rename columns
        df = df.join(df_temp)

    return df


def min_max_scaler(data):
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


def make_features(start_date, end_date, input_days=3, scaler='standard', is_training=False):
    # TODO: select symbols
    # commodity : BrentOil, Copper, CrudeOil, Gasoline, Gold, NaturalGas, Platinum, Silver
    # currency : AUD, CNY, EUR, GBP, HKD, JPY, USD
    # table = merge_data(start_date, end_date, symbols=['Gold', 'Silver','USD'])

    # TODO: cleaning or filling missing value
    # table.dropna(inplace=True)

    # TODO: select columns to use
    # gold_price = table['Gold_Price']

    # TODO: make features
    all_features_df, gold_price = create_all_features(start_date, end_date, is_training)
    features_top20 = ['Gold_volatility_kcp',
 'Platinum_momentum_stoch',
 'AUD_trend_macd_signal',
 'Gold_others_cr',
 'Gold_Change',
 'Gold_trend_sma_slow',
 'Gold_trend_macd',
 'Gold_momentum_roc',
 'CrudeOil_momentum_roc',
 'NaturalGas_others_cr',
 'GBP_momentum_stoch_signal',
 'BrentOil_momentum_roc',
 'AUD_Change',
 'Gold_trend_kst_diff',
 'Platinum_Change',
 'Copper_trend_cci',
 'Silver_Change',
 'Gold_volatility_kcw',
 'Gold_trend_dpo',
 'NaturalGas_volatility_kcw',
                     ]

    all_features_df = all_features_df[all_features_df.columns[all_features_df.columns.isin(features_top20)]]

    gold_diff = np.diff(gold_price)
    gold_price = gold_price[1:]

    input_days = input_days
    training_sets = list()
    for time in range(len(gold_price) - input_days):
        diff = gold_diff[time:time + input_days]
        price = gold_price[time:time + input_days]
        features = all_features_df[time:time + input_days].values.ravel(order='C')

        daily_feature = np.concatenate((diff[::-1], price, features))
        training_sets.append(daily_feature)

    training_x = training_sets[:-10]
    test_x = training_sets[-10:]

    if scaler == 'standard':
        scaler = StandardScaler()
        training_x = scaler.fit_transform(training_x)
        test_x = scaler.transform(test_x)
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
        training_x = scaler.fit_transform(training_x)
        test_x = scaler.transform(test_x)

    past_price = gold_price[-11:-1]
    target_price = gold_price[-10:]

    return training_x if is_training else (test_x, past_price, target_price)


def make_features_for_tuning(all_features_df, gold_price, n_clusters, input_days=3):
    # TODO: select symbols
    # commodity : BrentOil, Copper, CrudeOil, Gasoline, Gold, NaturalGas, Platinum, Silver
    # currency : AUD, CNY, EUR, GBP, HKD, JPY, USD
    # table = merge_data(start_date, end_date, symbols=['Gold', 'Silver','USD'])

    # TODO: cleaning or filling missing value
    # table.dropna(inplace=True)

    # TODO: select columns to use
    # gold_price = table['Gold_Price']

    # make features
    clustered_results = KMeans(n_clusters=n_clusters).fit(all_features_df.transpose())
    selected_features = get_selected_columns(clustered_results)
    selected_features_df = all_features_df.iloc[:, selected_features]

    gold_diff = np.diff(gold_price)
    gold_price = gold_price[1:]

    input_days = input_days
    training_sets = list()
    for time in range(len(gold_price)-input_days):
        diff = gold_diff[time:time + input_days]
        price = gold_price[time:time + input_days]
        features = all_features_df[time:time + input_days].values.ravel(order='C')

        # daily_feature = np.concatenate((diff[::-1], price))
        # 이전 금 값, 변화량, related features
        daily_feature = np.concatenate((diff[::-1], price, features))
        training_sets.append(daily_feature)

    x_scaler = MinMaxScaler()
    x_scaler.fit(training_sets)
    training_sets = x_scaler.transform(training_sets)

    training_x = training_sets[:-10]
    test_x = training_sets[-10:]

    past_price = gold_price[-11:-1]
    target_price = gold_price[-10:]

    return training_x, test_x, past_price, target_price, list(selected_features_df.columns)


def create_all_features(start_date, end_date, is_training=False):
    if is_training and os.path.exists('features/preprocessd_features.pkl'):
        all_features_df = pd.read_pickle('features/preprocessd_features.pkl')
    else:
        # currency_symbols = ['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD']
        commodity_symbols = ['BrentOil', 'Copper', 'CrudeOil', 'Gasoline', 'NaturalGas', 'Platinum', 'Silver']
        currency_symbols = ['AUD','CNY','EUR','GBP','HKD','JPY', 'USD']

        currency_df = merge_data(start_date, end_date, symbols=currency_symbols)
        currency_df = currency_df[[col for col in currency_df.columns if 'Gold' not in col]]
        currency_df.dropna(inplace=True)

        commodity_df = merge_data(start_date, end_date, symbols=commodity_symbols)
        commodity_df = commodity_df[[col for col in commodity_df.columns if 'Gold' not in col]]
        commodity_df = commodity_df[[col for col in commodity_df.columns if 'Volume' not in col]]
        commodity_df.dropna(inplace=True)

        gold_df = merge_data(start_date, end_date, symbols=['Gold'])
        gold_df = gold_df[[col for col in gold_df.columns if 'Volume' not in col]]
        gold_df.dropna(inplace=True)

        total_table = pd.concat([currency_df, commodity_df, gold_df], axis=1)
        total_table = total_table.fillna(method='ffill').dropna()

        all_features_df = total_table
        for sym in currency_symbols + commodity_symbols:
            add_trend_ta(all_features_df, sym + '_High', sym + '_Low', sym + '_Price', fillna=True, colprefix=sym+'_')
            add_volatility_ta(all_features_df, sym + '_High', sym + '_Low', sym + '_Price', fillna=True, colprefix=sym+'_')
            add_momentum_ta(all_features_df, sym+'_High', sym+'_Low', sym+'_Price', sym+'_Vol', fillna=True, colprefix=sym+'_')
            add_others_ta(all_features_df, sym + '_Price', fillna=True, colprefix=sym+'_')

        all_features_df.to_pickle('features/preprocessd_features.pkl')
    # print('preprocessed features: {}'.format(all_features_df.shape))

    max_num_columns = all_features_df.shape[1]

    gold_price = all_features_df['Gold_Price']
    features_list = list(all_features_df.columns)
    features_list.remove('Gold_Price')

    all_features_df = all_features_df[features_list]
    # scaler = MinMaxScaler()
    # scaler.fit(all_features_df)
    # scaler.transform()
    # all_features_df = min_max_scaler(all_features_df)

    return all_features_df, gold_price
