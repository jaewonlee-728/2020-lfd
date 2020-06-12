import pickle
import numpy as np
import pandas as pd
from DataGenerator import get_data_path, make_features
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from utils import DataGenerator_for_correlation_based, DataGenerator_select_features_RF, DataGenerator_KMeansClustering


def get_test_dollar_price(start_date, end_date):
    """
    Do not fix this function
    """
    df = pd.read_csv(get_data_path('USD'), index_col="Date", parse_dates=True, na_values=['nan'])
    price = df['Price'].loc[end_date: start_date][:10][::-1]
    return price


def predict_from_correlation_based(start_date, end_date):
    test_x, test_y = DataGenerator_for_correlation_based.make_features(start_date, end_date, is_training=False)

    ###################################################################################################################
    # inspect test data
    assert test_y.tolist() == get_test_dollar_price(start_date, end_date).tolist(), 'your test data is wrong!'
    ###################################################################################################################

    # TODO: fix pickle file name
    filename = 'model/model_based_correlation.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    # print('load complete')
    # print(loaded_model.get_params())

    predict = loaded_model.predict([test_x])
    # results_mae = mean_absolute_error(np.reshape(predict, -1), [float(y) for y in test_y])
    results_mae = mean_absolute_error(np.reshape(predict, -1), test_y)
    # print('mae: ', results_mae)

    return predict, results_mae


def predict_from_random_forest(start_date, end_date):
    input_days = 30
    # RF_folder = 'RF'
    # mlp_folder = 'mlp_model'
    n_estimators = 200
    y_scaling = False
    total_table = DataGenerator_select_features_RF.make_dataset(start_date, end_date)
    RF = RandomForestRegressor(n_estimators=n_estimators, n_jobs=16, random_state=0,
                               verbose=False)
    modified_set, importance_series, RF_path = DataGenerator_select_features_RF.RF_feature_selection(RF, total_table,
                                                                                                     input_days)

    test_x, test_y, x_scaler, y_scaler = DataGenerator_select_features_RF.data_split(modified_set, input_days,
                                                                                     y_scaling,
                                                                                     is_training=False)

    ###################################################################################################################
    # inspect test data
    assert test_y.tolist() == get_test_dollar_price(start_date, end_date).tolist(), 'your test data is wrong!'
    ###################################################################################################################

    filename = 'model/%s' % ('RF_mlp.model')
    model = pickle.load(open(filename, 'rb'))

    if y_scaler is None:
        pred = model.predict(x_scaler.transform(test_x))
    else:
        pred = model.predict(x_scaler.transform(test_x))
        pred = y_scaler.inverse_transform(pred)

    results_mae = mean_absolute_error(pred, [test_y])
    # print('mae: ', results_mae)

    return pred, results_mae


def predict_from_clustering(start_date, end_date):
    features_df, USD_price, _ = DataGenerator_KMeansClustering.make_raw_features(start_date, end_date, reuse=False)

    # load selected features
    with open('features/selected_features_from_clustering.txt', 'r', encoding='utf-8') as f:
        for index, line in enumerate(f.readlines()):
            if index == 1:
                selected_features = [x.replace('\n','') for x in line.split(", ")]
            # elif index == 0:
            #     print(line)
    f.close()

    selected_features_df = features_df[selected_features]

    x = DataGenerator_KMeansClustering.windowing_x(selected_features_df, 10)
    y = DataGenerator_KMeansClustering.windowing_y(USD_price, 10)

    test_x = x[-10]
    test_y = y[-10]

    ###################################################################################################################
    # inspect test data
    assert test_y.tolist() == get_test_dollar_price(start_date, end_date).tolist(), 'your test data is wrong!'
    ###################################################################################################################

    filename = 'model/model_kmeans_clustering_best.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    # print('load complete')
    # print(loaded_model.get_params())

    predict = loaded_model.predict([test_x])
    mae_results = mean_absolute_error(np.reshape(predict, -1), test_y)
    # print('mae: ', mae_results)

    return predict, mae_results



def main():
    start_date = '2010-01-01'
    end_date = '2020-04-06'

    print("\tLoad indivdual model with different feature selection methods")
    results_rf, mae_rf = predict_from_random_forest(start_date, end_date)
    print("random forest-based mae: ", mae_rf)
    results_clustering, mae_clustering = predict_from_clustering(start_date, end_date)
    print("clustering-based mae: ", mae_clustering)
    results_correlation, mae_correlation = predict_from_correlation_based(start_date, end_date)
    print("correlation-based mae: ", mae_correlation)

    # Simple average
    simple_ensemble_predictions = (np.reshape(results_correlation, -1)+np.reshape(results_clustering, -1)+np.reshape(results_rf, -1))/3.0

    # Weighted average
    data = [np.reshape(results_correlation, -1), np.reshape(results_clustering, -1), np.reshape(results_rf, -1)]
    total = (1/mae_correlation) + (1/mae_clustering) + (1/mae_rf)
    weights = [1/(total*mae_correlation), 1/(total*mae_clustering), 1/(total*mae_rf)]
    weighted_ensemble_predictions = np.average(data, axis=0, weights=weights)

    test_x, test_y = make_features(start_date, end_date, is_training=False)

    ###################################################################################################################
    # inspect test data
    assert test_y.tolist() == get_test_dollar_price(start_date, end_date).tolist(), 'your test data is wrong!'
    ###################################################################################################################

    print('simple ensemble mae: ', mean_absolute_error(simple_ensemble_predictions, test_y))
    print('final mae: ', mean_absolute_error(weighted_ensemble_predictions, test_y))


if __name__ == '__main__':
    main()
