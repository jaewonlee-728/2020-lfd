# -*- coding: utf-8 -*-

#== Packages =======================================================================
from .technical_indicator import *
import pandas as pd
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


#== Functions =======================================================================
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


def compile_data(start_date, end_date, features):
    # TODO: select symbols
    # commodity : BrentOil, Copper, CrudeOil, Gasoline, Gold, NaturalGas, Platinum, Silver
    # currency : AUD, CNY, EUR, GBP, HKD, JPY, USD
    table = merge_data(start_date, end_date, symbols=features)
    table.dropna(inplace=True)
    table = table.drop(columns=[c for c in table.columns if c.endswith('Volume')])
    for f in features:
        table = add_trend_ta(table, high=f+'_High', low=f+'_Low', close=f+'_Price')
        table = add_volatility_ta(table, high=f+'_High', low=f+'_Low', close=f+'_Price')
        table = add_others_ta(table, close=f+'_Price')
    table.dropna(axis='columns', inplace=True)
    return table


# Correlation
def corr_vis(data):
    corr = data.corr()
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        annot=True,
        annot_kws={"fontsize":6},
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )


def imp_df(column_names, importances):
    '''
    function for creating a feature importance dataframe
    '''
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df


# plotting a feature importance dataframe (horizontal barchart)
def var_imp_plot(imp_df, title):
    imp_df.columns = ['feature', 'feature_importance']
    sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df, orient = 'h', color = 'royalblue') \
       .set_title(title, fontsize = 20)
       


#== Correlation Visualization =========================================================
start_date = '2010-01-01'
end_date = '2020-04-06'

df = compile_data(start_date, end_date,
                  ['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD']+['BrentOil', 'Copper', 'CrudeOil', 'Gasoline', 'Gold', 'NaturalGas', 'Platinum', 'Silver'])

df.drop(columns=[c for c in df.columns if not c.endswith('_Price')], inplace=True)
corr_vis(df)


#== Random Forest Visualization =========================================================
table = compile_data(start_date, end_date,
                     # features with absolute correlation higher than 0.3
                     ['Gasoline', 'Copper', 'NaturalGas', 'BrentOil', 'CrudeOil', 'HKD', 'Platinum', 'CNY', 'USD'])

x = table.drop(columns='USD_Price')

# Get buy/sell signals
y = np.where(table['USD_Price'].shift(-1) > table['USD_Price'], 1, -1)

# Split to training and test datasets
idx = int(len(table)*0.7//10) #10
training_x = x[:-idx]
training_y = y[:-idx]
test_x = x[-idx:]
test_y = y[-idx:]

# RF
clf = RandomForestClassifier(random_state=1024)

# Create the model on train dataset
model = clf.fit(training_x, training_y)
print('Correct Prediction (%): ', accuracy_score(test_y, model.predict(test_x), normalize=True)*100.0)

# Print feature importance scores
base_imp = imp_df(training_x.columns, clf.feature_importances_)

# Visualize
var_imp_plot(base_imp.loc[:25], 'Default feature importance (scikit-learn)')

# Which features?
print([i for i in base_imp.feature[:65]])


