from DataGenerator import merge_data, symbol_dict
import numpy as np
import pandas as pd


# volatility 체크
def main():
    start_date, end_date = '2010-01-01', '2020-05-19'

    feature_company_list = ['cell', 'hmotor', 'naver', 'lgchem', 'lghnh', 'samsung1', 'sdi', 'sk', 'kospi']
    symbol_list = [symbol_dict[c] for c in feature_company_list]

    table = merge_data(start_date, end_date, symbol_list)

    result_df = pd.DataFrame(index=symbol_list, columns=['cumulative_return', 'volatility', 'average'])
    for symbol in symbol_list:
        symbol_df = table.filter(like=symbol)
        symbol_df.loc[:, 'perc_ret'] = (symbol_df[symbol+'_close']-symbol_df[symbol+'_open'])/symbol_df[symbol+'_open']
        symbol_df.loc[:, 'cumulative_return'] = np.exp(np.log1p(symbol_df['perc_ret']).cumsum())

        result_df.loc[symbol]['cumulative_return'] = symbol_df['cumulative_return'].tail(1).values[0]
        result_df.loc[symbol]['volatility'] = symbol_df['perc_ret'].std()
        result_df.loc[symbol]['average'] = symbol_df['perc_ret'].mean()

    print(result_df.sort_values(by='volatility'))


if __name__ == "__main__":
    main()


    # cumulative return 을 구하자