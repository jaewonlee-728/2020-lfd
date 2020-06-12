import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .technical_indicator import * # my code
import numpy as np # my code


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
    #df = df.dropna(subset=df.columns, how='all')
    return df


def make_features(start_date, end_date, is_training):
    # TODO: select symbols
    # commodity : BrentOil, Copper, CrudeOil, Gasoline, Gold, NaturalGas, Platinum, Silver
    # currency : AUD, CNY, EUR, GBP, HKD, JPY, USD
    features = ['Gasoline', 'Copper', 'NaturalGas', 'BrentOil', 'CrudeOil', 'HKD', 'Platinum', 'CNY', 'USD'] # features with absolute correlation higher than 0.3
    table = merge_data(start_date, end_date, symbols=features)
    table.dropna(inplace=True)
    
    # TODO: cleaning or filling missing value
    table = table.drop(columns=[c for c in table.columns if c.endswith('Volume')])

    # TODO: select columns to use
    USD_price = table['USD_Price']

    # TODO: make your features
    
    for f in features:
        table = add_trend_ta(table, high=f+'_High', low=f+'_Low', close=f+'_Price')
        table = add_volatility_ta(table, high=f+'_High', low=f+'_Low', close=f+'_Price')
        table = add_others_ta(table, close=f+'_Price')
    
    table.dropna(axis='columns', inplace=True)
    
    df = table#[['Platinum_Change', 'Copper_Change', 'NaturalGas_Change', 'trend_dpo', 'Gasoline_Change', 'trend_adx_pos', 'trend_kst_diff', 'others_dr', 'CrudeOil_Change', 'HKD_Change', 'CNY_Change', 'volatility_kcp', 'USD_Change', 'trend_adx', 'volatility_bbw', 'trend_mass_index', 'BrentOil_Change', 'volatility_kcw', 'trend_adx_neg', 'volatility_atr', 'trend_kst', 'NaturalGas_High', 'trend_kst_sig', 'Platinum_Price', 'NaturalGas_Open', 'others_cr', 'CNY_Price', 'NaturalGas_Price', 'CNY_High', 'trend_psar', 'HKD_Low', 'Copper_High', 'Platinum_High', 'HKD_Price', 'CNY_Open', 'Platinum_Open', 'CNY_Low', 'USD_Open', 'Copper_Low', 'HKD_High', 'CrudeOil_Low', 'NaturalGas_Low', 'volatility_bbl', 'CrudeOil_Open', 'USD_Low', 'Copper_Open', 'BrentOil_Price', 'CrudeOil_Price', 'HKD_Open', 'Gasoline_Low', 'volatility_kcc', 'USD_High', 'trend_visual_ichimoku_a', 'Gasoline_Open', 'Gasoline_High', 'Platinum_Low', 'Copper_Price', 'volatility_bbh', 'Gasoline_Price', 'BrentOil_Open', 'BrentOil_Low', 'BrentOil_High', 'CrudeOil_High', 'volatility_kch', 'trend_aroon_ind']]

    input_days = 50

    x = windowing_x(df, input_days)
    y = windowing_y(USD_price, input_days)

    # split training and test data

    training_x = x[:-10]
    training_y = y[:-10]
    test_x = x[-10]
    test_y = y[-10]
    
    scaler = MinMaxScaler()
    training_x = scaler.fit_transform(training_x)
    test_x = scaler.transform(test_x.reshape(1,df.shape[1]*input_days)).reshape(df.shape[1]*input_days,)
    
    return (training_x, training_y) if is_training else (test_x, test_y)


def windowing_y(data, input_days):
    windows = [data[i + input_days:i + input_days + 10] for i in range(len(data) - input_days)]
    return windows


def windowing_x(data, input_days):
    windows = np.array([data[i:i + input_days].values.ravel(order='C') for i in range(len(data) - input_days)])
    return windows


if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2020-04-06'

    make_features(start_date, end_date, is_training=False)

