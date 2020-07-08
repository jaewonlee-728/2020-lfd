from matplotlib import pyplot as plt

from sklearn import svm
import DataGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import resample
import pandas as pd
import pickle


def f_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()



test_day = ['2020-01-19', '2020-02-01', '2020-02-02', '2020-02-08']
# test_day = ['2020-02-09', '2020-02-15', '2020-02-16', '2020-02-22']
training_x, training_y = DataGenerator.get_data(test_day, is_training=True)

# Up-sample
training_df = pd.concat([training_x, training_y], axis=1)

minor_df = training_df[training_df.win == 1]
major_df = training_df[training_df.win == 0]

minor_df_upsample = resample(minor_df, replace=True, n_samples=len(major_df), random_state=1)

new_training_df = pd.concat([major_df, minor_df_upsample], axis=0)

training_y = new_training_df['win']
training_x = new_training_df.drop(['win'], axis=1)

scaler = StandardScaler()
new_training_x = scaler.fit_transform(training_x, training_y)

# TODO: fix pickle file name
filename = 'team00_model_jw.pkl'
model = pickle.load(open(filename, 'rb'))
print('load complete')
print(model.get_params())

f_importances(model.coef_[0], list(new_training_df.columns))