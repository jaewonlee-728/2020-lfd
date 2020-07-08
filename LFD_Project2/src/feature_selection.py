import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
import DataGenerator as dg
import numpy as np

start_date = '2010-01-01'
end_date = '2020-04-18'
is_training = True

# define dataset
all_features_df, gold_price = dg.create_all_features(start_date, end_date, is_training)
all_features_df=all_features_df[all_features_df.columns[~all_features_df.columns.str.contains('_Price')]]
all_features_df=all_features_df[all_features_df.columns[~all_features_df.columns.str.contains('_Open')]]
all_features_df=all_features_df[all_features_df.columns[~all_features_df.columns.str.contains('_High')]]
all_features_df=all_features_df[all_features_df.columns[~all_features_df.columns.str.contains('_Low')]]
all_features_df=all_features_df[all_features_df.columns[~all_features_df.columns.str.contains('_adx')]]
all_features_df=all_features_df[all_features_df.columns[~all_features_df.columns.str.contains('_vortex')]]
all_features_df=all_features_df[1:]

index=list(all_features_df.columns)
all_features_df=all_features_df[:-1]

x_scaler = MinMaxScaler()
x_scaler.fit(all_features_df.values)
X = x_scaler.transform(all_features_df.values)

gold_diff=np.diff(gold_price)
gold_diff=gold_diff[1:]
y_scaler = MinMaxScaler()
y_scaler.fit(gold_diff.reshape(-1,1))
y = y_scaler.transform(gold_diff.reshape(-1,1))
y=y.reshape(-1)

importance_ranking=pd.DataFrame({'factor':index})

#####RandomForest
# define the model
model = RandomForestRegressor()
# fit the model
model.fit(X, y)
# get importance
importance_ranking['rf_imp'] = model.feature_importances_
importance_ranking['rf_imp_rank']=importance_ranking['rf_imp'].rank(ascending=False)

#####XGBoost
# define the model
model = XGBRegressor()
# fit the model
model.fit(X, y)
# get importance
importance_ranking['xgb_imp'] = model.feature_importances_
importance_ranking['xgb_imp_rank'] = importance_ranking['xgb_imp'].rank(ascending=False)

#####Ensemble Ranking
rank_columns=importance_ranking.columns[importance_ranking.columns.str.contains('_rank')]
importance_ranking['average_rank']=importance_ranking[rank_columns].mean(axis=1)
importance_ranking['overall_rank']=importance_ranking['average_rank'].rank()
importance_ranking=importance_ranking.sort_values(by='overall_rank')
importance_ranking=importance_ranking.reset_index(drop=True)

importance_ranking.to_csv('./features/importance_ranking.csv')

importance_ranking.head(20)['factor'].to_list()
