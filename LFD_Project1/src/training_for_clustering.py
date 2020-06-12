import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn import neural_network as NN
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from utils import DataGenerator_KMeansClustering
import pickle
import os

def clustering_for_features_selection(start_date, end_date):
    features_df, USD_price, max_num_features = DataGenerator_KMeansClustering.make_raw_features(start_date, end_date)

    # Use K-means for features selection
    n_clusters_list = list(range(10, max_num_features, 5))

    print('Start selecting features from {}'.format(n_clusters_list))
    mae_results = []
    results_file = open('features/results_features_selection_from_clustering.txt.txt', 'w', encoding='utf-8')
    for n_cluster in n_clusters_list:
        selected_features, training_x, training_y, test_x, test_y = DataGenerator_KMeansClustering.select_features(features_df,
                                                                                                         USD_price,
                                                                                                         n_cluster)
        print('{} {}'.format(n_cluster, ','.join(selected_features)))

        mlp = NN.MLPRegressor()
        mlp.fit(training_x, training_y)

        predict = mlp.predict([test_x])
        res_mae = mean_absolute_error(np.reshape(predict, -1), test_y)
        if not mae_results or min(mae_results) > res_mae:
            # Save features
            with open('features/selected_features_from_clustering.txt', 'w', encoding='utf-8') as f:
                f.write('{}, {}\n'.format(n_cluster, res_mae))
                f.write(', '.join(selected_features))
            f.close()

            # Save model
            # TODO: fix pickle file name
            filename = 'model/model_kmeans_clustering_best.pkl'.format(len(selected_features), str(res_mae))
            pickle.dump(mlp, open(filename, 'wb'))
            print('saved {}'.format(filename))

        mae_results.append(res_mae)
        print('mae for {} clusters: {}'.format(n_cluster, res_mae))
        results_file.write('mae for {} clusters: {}\n'.format(n_cluster, res_mae))

    results_file.close()

    plt.plot(n_clusters_list, mae_results, 'b-')
    plt.grid(which='both')
    plt.xticks(list(range(10, max(n_clusters_list), 50)))
    plt.yticks(list(range(0, int(max(mae_results)), 5)))
    # plt.axis([0, max(n_clusters_list), 0, max(mae_results)])
    plt.ylabel('MAE')
    plt.xlabel('number of clusters')
    plt.show()
    plt.savefig('features/results_features_selection_from_clustering.png')


def tuning_model_hyperparameters(start_date, end_date):
    features_df, USD_price, _ = DataGenerator_KMeansClustering.make_raw_features(start_date, end_date, reuse=True)

    # load selected features
    with open('features/selected_features_from_clustering.txt', 'r', encoding='utf-8') as f:
        for index, line in enumerate(f.readlines()):
            if index == 1:
                selected_features = [x.replace('\n','') for x in line.split(", ")]
            # elif index == 0:
            #     print(line)
    f.close()

    selected_features_df = features_df[selected_features]
    print(selected_features_df.shape)

    x = DataGenerator_KMeansClustering.windowing_x(selected_features_df, 10)
    y = DataGenerator_KMeansClustering.windowing_y(USD_price, 10)

    training_x = x[:-10]
    training_y = y[:-10]

    test_x = x[-10]
    test_y = y[-10]

    # Tuning starts from here...
    nhn_range = [(70, 50, 30, 10), (80, 60, 40, 20, 10), (100, 50, 25, 10),
                 (100, 100), (100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100, 100),
                 (50, 50), (50, 50, 50), (50, 50, 50, 50), (50, 50, 50, 50, 50)]  # number of hidden neurons
    result_list = [0]
    for nhn in nhn_range:
        # mlp = NN.MLPRegressor(hidden_layer_sizes=nhn,
        #                       max_iter=50000, solver='adam',
        #                       shuffle=True, activation='relu', learning_rate='constant',
        #                       early_stopping=True, batch_size='auto',
        #                       random_state=13, learning_rate_init=0.001,
        #                       n_iter_no_change=20, validation_fraction=0.2,
        #                       alpha=0.01, epsilon=1e-08,
        #                       verbose=False)
        mlp = NN.MLPRegressor(hidden_layer_sizes=nhn)

        mlp.fit(training_x, training_y)
        predict = mlp.predict([test_x])
        res_mae = mean_absolute_error(np.reshape(predict, -1), test_y)
        print(nhn, res_mae)

        if not result_list or min(result_list) > res_mae:
            filename = 'model/model_kmeans_clustering_best_2.pkl'
            pickle.dump(mlp, open(filename, 'wb'))
            print('saved {} {}'.format(filename, res_mae))
        result_list.append(res_mae)


def training(start_date, end_date):
    features_df, USD_price, _ = DataGenerator_KMeansClustering.make_raw_features(start_date, end_date, reuse=True)

    # load selected features
    with open('features/selected_features_from_clustering.txt', 'r', encoding='utf-8') as f:
        for index, line in enumerate(f.readlines()):
            if index == 1:
                selected_features = [x.replace('\n', '') for x in line.split(", ")]
            # elif index == 0:
            #     print(line)
    f.close()

    selected_features_df = features_df[selected_features]
    print(selected_features_df.shape)

    x = DataGenerator_KMeansClustering.windowing_x(selected_features_df, 10)
    y = DataGenerator_KMeansClustering.windowing_y(USD_price, 10)

    training_x = x[:-10]
    training_y = y[:-10]

    test_x = x[-10]
    test_y = x[-10]

    if os.path.exists('model/model_kmeans_clustering_best.pkl'):
        mlp = pickle.load('model/model_kmeans_clustering_best.pkl')
    else:
        mlp = NN.MLPRegressor(
            max_iter=50000, solver='adam',
            shuffle=True, activation='relu', learning_rate='constant',
            early_stopping=True, batch_size='auto',
            random_state=13, learning_rate_init=0.001,
            n_iter_no_change=20, validation_fraction=0.2,
            alpha=0.01, epsilon=1e-08,
            verbose=False)

        mlp.fit(training_x, training_y)
        filename = 'model/model_kmeans_clustering_final.pkl'
        pickle.dump(mlp, open(filename, 'wb'))
        print('saved {}'.format(filename))

    predict = mlp.predict([test_x])
    res_mae = mean_absolute_error(np.reshape(predict, -1), test_y)

    print(res_mae)


# To find best models with features
def main():
    start_date = '2010-01-01'
    end_date = '2020-04-06'

    clustering_for_features_selection(start_date, end_date)
    tuning_model_hyperparameters(start_date, end_date)


if __name__ == "__main__":
    main()





