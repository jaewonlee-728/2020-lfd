from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import DataGenerator
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import svm


test_day = ['2020-01-19', '2020-02-01', '2020-02-02', '2020-02-08']
# test_day = ['2020-02-09', '2020-02-15', '2020-02-16', '2020-02-22']
training_x, training_y = DataGenerator.get_data(test_day, is_training=True)

# Up-sample
training_df = pd.concat([training_x, training_y], axis=1)

minor_df = training_df[training_df.win==1]
major_df = training_df[training_df.win==0]

minor_df_upsample = resample(minor_df, replace=True, n_samples=len(major_df), random_state=1)

new_training_df = pd.concat([major_df, minor_df_upsample], axis=0)

training_y = new_training_df['win']
training_x = new_training_df.drop(['win'], axis=1)

scaler = MinMaxScaler()
training_x = scaler.fit_transform(training_x, training_y)

model = svm.SVC(C=1, kernel='linear', random_state=0)

feature_selection = SFS(model, forward=False, cv=10, k_features=(5, 15), scoring='f1', verbose=True, n_jobs=-1)
feature_selection.fit(training_x, training_y)

print(f"Best score achieved: {feature_selection.k_score_}, Feature names: {feature_selection.k_feature_names_}")
