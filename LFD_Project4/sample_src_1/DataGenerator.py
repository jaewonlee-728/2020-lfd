import os
import pandas as pd
import numpy as np

symbol_dict = {'cell': 'Celltrion',
               'hmotor': 'HyundaiMotor',
               'naver': 'NAVER',
               'lgchem': 'LGChemical',
               'lghnh': 'LGH&H',
               'bio': 'SamsungBiologics',
               'samsung1': 'SamsungElectronics',
               'samsung2': 'SamsungElectronics2',
               'sdi': 'SamsungSDI',
               'sk': 'SKhynix',
               'kospi': 'KOSPI'}


def symbol_to_path(symbol, base_dir="../data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def merge_data(start_date, end_date, symbols):
    dates = pd.date_range(start_date, end_date)

    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date", parse_dates=True,
                              usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Open': symbol + '_open', 'High': symbol + '_high', 'Low': symbol + '_low',
                                          'Close': symbol + '_close', 'Volume': symbol + '_volume'})
        df = df.join(df_temp)

    # TODO: cleaning or filling missing value
    df = df.dropna()

    return df


def make_features(trade_company_list, start_date, end_date, is_training):

    # TODO: Choose symbols to make feature
    # symbol_list = ['Celltrion', 'HyundaiMotor', 'NAVER', 'LGChemical', 'LGH&H', 'SamsungBiologics',
    #                 'SamsungElectronics', 'SamsungElectronics2', 'SamsungSDI', 'SKhynix', 'KOSPI']
    feature_company_list = ['cell', 'lgchem', 'samsung1', 'kospi']
    symbol_list = [symbol_dict[c] for c in feature_company_list]

    table = merge_data(start_date, end_date, symbol_list)

    # DO NOT CHANGE
    test_days = 10
    open_prices = np.asarray(table[[symbol_dict[c] + '_open' for c in trade_company_list]])
    close_prices = np.asarray(table[[symbol_dict[c] + '_close' for c in trade_company_list]])

    # TODO: select columns to use
    data = dict()
    for c in feature_company_list:
        data[c, 'close'] = table[symbol_dict[c] + '_close']
        data[c, 'open'] = table[symbol_dict[c] + '_open']
        data[c, 'open_ema'] = table[symbol_dict[c] + '_open'].ewm(alpha=0.5).mean()

    # TODO: make features
    input_days = 3

    features = list()
    for a in range(data['kospi', 'close'].shape[0] - input_days):

        # kospi close price
        kospi_close_feature = data['kospi', 'close'][a:a + input_days]

        # stock close price : samsung1
        tmps = list()
        for c in trade_company_list:
            tmp = data[c, 'close'][a:a + input_days]
            tmps.append(tmp)
        close_feature = np.concatenate(tmps, axis=0)

        # stock open ema price : cell, lgchem, samsung1
        tmps = list()
        for symbol in ['cell', 'lgchem', 'samsung1']:
            tmp = data[symbol, 'open_ema'][a:a + input_days]
            tmps.append(tmp)
        ema_feature = np.concatenate(tmps, axis=0)

        features.append(np.concatenate([
                                        kospi_close_feature,
                                        close_feature,
                                        ema_feature,
                                        ], axis=0))

    if not is_training:
        return open_prices[-test_days:], close_prices[-test_days:], features[-test_days:]

    return open_prices[input_days:], close_prices[input_days:], features


if __name__ == "__main__":
    trade_company_list = ['samsung1']
    open, close, feature = make_features(trade_company_list, '2010-01-01', '2019-05-08', False)
    print(open.T[0],'\n')
    print(close.T[0],'\n')
    print(*feature[0],sep=' / ')
