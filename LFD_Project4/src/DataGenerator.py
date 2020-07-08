import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

symbol_dict = {'cell': 'Celltrion',
               'hmotor': 'HyundaiMotor',
               'naver': 'NAVER',
               'kakao': 'Kakao',
               'lgchem': 'LGChemical',
               'lghnh': 'LGH&H',
               'bio': 'SamsungBiologics',
               'samsung1': 'SamsungElectronics',
               'samsung2': 'SamsungElectronics2',
               'sdi': 'SamsungSDI',
               'sk': 'SKhynix',
               'kospi': 'KOSPI', }


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
        if symbol == 'NAVER':
            stock_split_date_index = df_temp.index.get_loc(pd.to_datetime('2018-10-12'), method='nearest')

            before_stock_split_df = df_temp.iloc[:stock_split_date_index]
            before_stock_split_df = before_stock_split_df.replace(to_replace=0, method='ffill')
            before_stock_split_df.loc[:, before_stock_split_df.columns != 'NAVER_volume'] =\
                before_stock_split_df.loc[:, before_stock_split_df.columns != 'NAVER_volume']/5
            after_stock_split_df = df_temp.iloc[stock_split_date_index:]

            df_temp = pd.concat([before_stock_split_df, after_stock_split_df], axis=0)

        df = df.join(df_temp)

    # TODO: cleaning or filling missing value
    df = df.dropna()

    # KOSPI_volume 열의 형태 바꾸기 (예: 296,548K —> 296548000)
    if 'KOSPI' in symbols:
        df['KOSPI_volume'] = df['KOSPI_volume'].apply(lambda x: float(x.replace(',', '').replace('K', '') + '000'))

    return df


def rsi(df, period):
    U = np.where(df.diff(1) > 0, df.diff(1), 0)
    D = np.where(df.diff(1) < 0, df.diff(1) * (-1), 0)
    AU = pd.DataFrame(U, index=df.index).rolling(window=period).mean()
    AD = pd.DataFrame(D, index=df.index).rolling(window=period).mean()
    RSI = AU / (AD + AU) * 100
    return RSI


def macd(m_Df, m_NumFast=12, m_NumSlow=26, m_NumSignal=9):
    for col in m_Df:
        if 'close' in col:
            comp_name = col.split('_close')[0]
            m_Df[comp_name + '_EMAFast'] = m_Df[comp_name + '_close'].ewm(span=m_NumFast,
                                                                          min_periods=m_NumFast - 1).mean()
            m_Df[comp_name + '_EMASlow'] = m_Df[comp_name + '_close'].ewm(span=m_NumSlow,
                                                                          min_periods=m_NumSlow - 1).mean()
            m_Df[comp_name + '_MACD'] = m_Df[comp_name + '_EMAFast'] - m_Df[comp_name + '_EMASlow']
            m_Df[comp_name + '_MACDSignal'] = m_Df[comp_name + '_MACD'].ewm(span=m_NumSignal,
                                                                            min_periods=m_NumSignal - 1).mean()
            m_Df[comp_name + '_MACDDiff'] = m_Df[comp_name + '_MACD'] - m_Df[comp_name + '_MACDSignal']
    return m_Df


def make_features(trade_company_list, start_date, end_date, is_training):
    # TODO: Choose symbols to make feature
    # symbol_list = ['Celltrion', 'HyundaiMotor', 'NAVER', 'Kakao', 'LGChemical', 'LGH&H',
    #                 'SamsungElectronics', 'SamsungElectronics2', 'SamsungSDI', 'SKhynix', 'KOSPI']
    feature_company_list = ['cell', 'hmotor', 'naver', 'lgchem', 'lghnh', 'samsung1', 'sdi', 'sk', 'kakao', 'kospi']

    # kospi는 가장 volatility가 낮아서 buffer 역할 수행

    symbol_list = [symbol_dict[c] for c in feature_company_list]



    table = merge_data(start_date, end_date, symbol_list)
    for col in table.columns:
        if 'close' in col:
            comp_name = col.split('_close')[0]
            table[comp_name + '_rsi'] = rsi(table[col], 14)

    table = macd(table)
    # DO NOT CHANGE
    test_days = 10
    open_prices = np.asarray(table[[symbol_dict[c] + '_open' for c in trade_company_list]])
    close_prices = np.asarray(table[[symbol_dict[c] + '_close' for c in trade_company_list]])

    # TODO: select columns to use
    data = dict()

    for c in feature_company_list:
        data[c, 'close'] = table[symbol_dict[c] + '_close']
        # data[c, 'open'] = table[symbol_dict[c] + '_open'] # 종가, 시가 등은 서로 상관관계가 있으니 제거
        data[c, 'close_ema'] = table[symbol_dict[c] + '_close'].ewm(alpha=0.5).mean()
        data[c, 'rsi'] = table[symbol_dict[c] + '_rsi']
        data[c, 'macd'] = table[symbol_dict[c] + '_MACD']
        data[c, 'macd_diff'] = table[symbol_dict[c] + '_MACDDiff']
        data[c, 'close_pc'] = table[symbol_dict[c] + '_close'].pct_change().fillna(0)


    # TODO: make features
    input_days = 1

    features = list()
    for a in range(data['kospi', 'close'].shape[0] - input_days):

        # kospi close price
        kospi_close_feature = data['kospi', 'close'][a:a + input_days]

        # stock close price: cell, sk, kakao
        tmps = list()
        for c in trade_company_list:
            tmp = data[c, 'close'][a:a + input_days]  # 시가 기준 정규화
            tmps.append(tmp)
        close_feature = np.concatenate(tmps, axis=0)

        # stock close ema price: cell, sk, kakao
        tmps = list()
        for symbol in feature_company_list:
            tmp = data[symbol, 'close_ema'][a:a + input_days]
            tmps.append(tmp)
        ema_feature = np.concatenate(tmps, axis=0)

        # 추가 코드: stock close rsi: cell, sk, kakao
        tmps = list()
        for symbol in feature_company_list:
            tmp = data[symbol, 'rsi'][a:a + input_days]
            tmps.append(tmp)
        rsi_feature = np.concatenate(tmps, axis=0)

        # macd
        tmps = list()
        for symbol in feature_company_list:
            tmp = data[symbol, 'macd'][a:a + input_days]
            tmps.append(tmp)
        macd_feature = np.concatenate(tmps, axis=0)

        # macd_dff
        tmps = list()
        for symbol in feature_company_list:
            tmp = data[symbol, 'macd_diff'][a:a + input_days]
            tmps.append(tmp)
        macdDiff_feature = np.concatenate(tmps, axis=0)

        tmps = list()
        for symbol in feature_company_list:
            tmp = data[symbol, 'close_pc'][a:a + input_days]
            tmps.append(tmp)
        pc_feature = np.concatenate(tmps, axis=0)

        features.append(np.concatenate([
            kospi_close_feature,
            close_feature,
            ema_feature,
            rsi_feature,  # 추가 코드: stock close rsi
            # rsi_volume_feature,# 추가 코드: volume rsi
            macd_feature,
            macdDiff_feature,
            pc_feature
        ], axis=0))
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    if not is_training:
        return open_prices[-test_days:], close_prices[-test_days:], features[-test_days:]

    return open_prices[input_days:], close_prices[input_days:], features


if __name__ == "__main__":
    trade_company_list = ['lgchem', 'samsung1']  # 실제로 trade할 company
    open, close, feature = make_features(trade_company_list, '2010-01-01', '2019-05-08', False)
    print(open, '\n')
    print(close, '\n')
    print(*feature[0], sep=' / ')