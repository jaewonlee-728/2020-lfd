import os
import pandas as pd


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


def make_features(start_date, end_date, is_training):
    # TODO: select symbols
    # commodity : BrentOil, Copper, CrudeOil, Gasoline, Gold, NaturalGas, Platinum, Silver
    # currency : AUD, CNY, EUR, GBP, HKD, JPY, USD
    table = merge_data(start_date, end_date, symbols=['AUD', 'CNY', 'EUR'])

    # TODO: cleaning or filling missing value
    table.dropna(inplace=True)

    # TODO: select columns to use
    USD_price = table['USD_Price']
    # USD_open = table['USD_Open']
    # USD_high = table['USD_High']
    # USD_low = table['USD_Low']

    # TODO: make your features
    input_days = 5

    x = windowing_x(USD_price, input_days)
    y = windowing_y(USD_price, input_days)

    # split training and test data
    training_x = x[:-10]
    training_y = y[:-10]
    test_x = x[-10]
    test_y = y[-10]

    return (training_x, training_y) if is_training else (test_x, test_y)


def windowing_y(data, input_days):
    windows = [data[i + input_days:i + input_days + 10] for i in range(len(data) - input_days)]
    return windows


def windowing_x(data, input_days):
    windows = [data[i:i + input_days] for i in range(len(data) - input_days)]
    return windows


if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2020-04-06'

    make_features(start_date, end_date, is_training=False)

