import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
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


def make_dataset(start_date, end_date) :
    currency_symbols = ['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY']
    commodity_symbols = ['BrentOil', 'Copper', 'CrudeOil', 'Gasoline', 'Gold', 'NaturalGas', 'Platinum', 'Silver']
    symbols = currency_symbols + commodity_symbols
    
    
    money_table = merge_data(start_date, end_date, symbols=currency_symbols)
    money_table = money_table[[col for col in money_table.columns if 'Change' not in col]]
    money_table.dropna(inplace=True)
    
    material_table = merge_data(start_date, end_date, symbols=commodity_symbols)
    material_table = material_table[[col for col in material_table.columns if 'Volume' not in col]]
    material_table = material_table[[col for col in material_table.columns if 'Change' not in col]]
    material_table.dropna(inplace=True)
    
    total_table = pd.concat([money_table, material_table.iloc[:,4:]], axis = 1)
    total_table = total_table.fillna(method = 'ffill').dropna()
    
    for sym in symbols :
        add_others_ta(total_table, sym+'_Price', fillna=True, colprefix = sym+'_')
        add_trend_ta(total_table, sym+'_High', sym+'_Low', sym+'_Price', fillna=True,
                        colprefix = sym+'_')
        add_volatility_ta(total_table, sym+'_High', sym+'_Low', sym+'_Price', fillna=True,
                             colprefix = sym+'_')
    return total_table


def RF_feature_selection(RF, total_table, input_days = 30, top_n = 40, RF_folder = 'RF', 
                         n_estimators = 200) :
    x = windowing_x(total_table, input_days)
    y = windowing_y(total_table['USD_Price'], input_days)
    
    RF_X = [i.values.ravel() for i in x]
    RF_Y = [np.mean(i)for i in y]
    
    RF_X_train = RF_X[:-10]
    RF_Y_train = RF_Y[:-10]
    RF_X_test = RF_X[-10]
    RF_Y_test = RF_Y[-10]
    path = 'model/RF_%sd_%sn.model' % (input_days, n_estimators)

    if os.path.exists(path) :
        with open(path,"rb") as fr:
            RF = pickle.load(fr)
    else :
        RF.fit(RF_X_train, RF_Y_train)
        with open(path,"wb") as fw:
            pickle.dump(RF, fw)
    # print(np.abs(RF.predict(RF_X_test.reshape(1,-1)) - RF_Y_test).mean())
    
    total_feature = []
    for num in range(input_days) :
        total_feature += [i+'_'+str(num+1) for i in total_table.columns]
    
    feature_table = pd.DataFrame(index=total_feature)
    feature_table['importance'] = RF.feature_importances_
    
    importance_series = feature_table.sort_values('importance', ascending=False)[:top_n]
    
    avg_importance = {}
    for i in total_table.columns :
        temp = [j for j in total_feature if i in j]
        avg_importance[i] = feature_table.loc[temp].mean().values[0]
        
    modified_set = total_table[list(map(lambda x : x[0],sorted(avg_importance.items())[-top_n:]))]
    return modified_set, importance_series, path


def data_split(modified_set, input_days, y_scaling, is_training=True) :
    x = windowing_x(modified_set, input_days)
    y = windowing_y(modified_set['USD_Price'], input_days)
    training_x, training_y = x[:-10], y[:-10]
    test_x, test_y = x[-10], y[-10]
    training_x = [i.values.ravel() for i in training_x]
    test_x = test_x.values.ravel().reshape(1,-1)
    test_y = test_y.values.reshape(1,-1)
    
    x_scaler = MinMaxScaler()
    x_scaler.fit(training_x)
    training_x = x_scaler.transform(training_x)
    y_scaler = None
    if y_scaling :
        y_scaler = MinMaxScaler()
        y_scaler.fit(training_y)
        training_y = y_scaler.transform(training_y)
    
    if is_training :
        return training_x, training_y, x_scaler, y_scaler
    else :
        return test_x, test_y[0], x_scaler, y_scaler


if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2020-04-06'

    make_features(start_date, end_date, is_training=False)

