from sklearn.metrics import mean_absolute_error
import numpy as np
import pickle
import DataGenerator
import pandas as pd
import os
import DataGenerator as dg
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
from hmmlearn.hmm import MultinomialHMM
import random


def k_cross_validation(k, start_date, end_date, input_days, scaler, n_components):
    error = 0
    for i in range(k):
        start_date = pd.to_datetime(start_date) - pd.Timedelta(days=time_step)
        current_end_date = pd.to_datetime(end_date) - pd.Timedelta(days=time_step * i)
        model = GaussianHMM(n_components=n_components, n_iter=100)
        training_x = dg.make_features(start_date, current_end_date, input_days=input_days, scaler=scaler,
                                      is_training=True)
        model.fit(training_x)

        test_x, past_price, target_price = dg.make_features(start_date, current_end_date, input_days=input_days,
                                                            scaler=scaler, is_training=False)
        hidden_states = model.predict(test_x)
        means = np.sum(model.means_, axis=1)
        expected_diff_price = np.dot(model.transmat_, model.means_)
        diff = list(zip(*expected_diff_price))[0]

        predicted_price = list()
        for idx in range(10):  # predict gold price for 10 days
            state = hidden_states[idx]
            current_price = past_price[idx]
            next_day_price = current_price + diff[state]  # predicted gold price of next day
            predicted_price.append(next_day_price)

        predict = np.array(predicted_price)
        print("\nk = {}".format(i))
        print('past price : {}'.format(np.array(past_price)))
        print('predicted price : {}'.format(predict))
        print('real price : {}'.format(np.array(target_price)))
        print('\tMAE : {}'.format(mean_absolute_error(target_price, predict)))
        error += mean_absolute_error(target_price, predict)
    return error / k


if __name__ == '__main__':
    start_date = '2010-01-01'
    end_date = '2020-04-18'
    n_components = [3,5,7]
    k = 20
    time_step = 10
    result_df = pd.DataFrame(columns=['scaler', 'input_days', 'MAE'])
    for n in n_components:
        for scaler in ['minmax', 'standard']:
            for input_days in [3, 5]:
                error = k_cross_validation(k, start_date, end_date, input_days, scaler, n)
                result_df = result_df.append({'scaler': scaler, 'input_days': input_days, 'components': n, 'MAE': error}, ignore_index=True)
                print("-------------------------------")
                print('Parameter : {}, {}, {} / MAE {}'.format(scaler, input_days, n, error))
                print("===============================")