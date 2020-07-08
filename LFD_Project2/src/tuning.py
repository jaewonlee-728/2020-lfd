import DataGenerator
from hmmlearn.hmm import GaussianHMM
import numpy as np
from sklearn.metrics import mean_absolute_error
import pickle
from DataGenerator import make_features_for_tuning, create_all_features
import matplotlib.pyplot as plt


def validate_model(model, test_x, past_price):
    hidden_states = model.predict(test_x)
    expected_diff_price = np.dot(model.transmat_, model.means_)
    diff = list(zip(*expected_diff_price))[0]

    predicted_price = list()
    for idx in range(10):  # predict gold price for 10 days
        state = hidden_states[idx]
        current_price = past_price[idx]
        next_day_price = current_price + diff[state]  # predicted gold price of next day

        predicted_price.append(next_day_price)

    predict = np.array(predicted_price)

    return predict


def clustering_for_features_selection(start_date, end_date):
    all_features_df, gold_price = create_all_features(start_date, end_date, is_training=False)

    n_components = 3  # TODO tuning
    input_days = 3  # TODO tuning
    n_clusters_list = list(range(10, len(all_features_df.columns), 50))
    print(n_clusters_list)

    results_file = open('features/clustering_features_selection_results.txt', 'w', encoding='utf-8')
    mae_results = []
    for n_cluster in n_clusters_list:
        training_x, test_x, past_price, target_price, selected_features_name_list = make_features_for_tuning(
            all_features_df, gold_price, n_cluster, input_days)

        model = GaussianHMM(n_components)
        model.fit(training_x)

        predict = validate_model(model, test_x, past_price)
        res_mae = mean_absolute_error(target_price, predict)

        # print predicted_prices
        # print('past price : {}'.format(np.array(past_price)))
        # print('predicted price : {}'.format(predict))
        # print('real price : {}'.format(np.array(target_price)))
        # print()
        # print('mae :', mean_absolute_error(target_price, predict))

        if not mae_results or min(mae_results) > res_mae:
            # Save features
            with open('features/clustering_selected_features.txt', 'w', encoding='utf-8') as f:
                f.write('{}, {}\n'.format(n_cluster, res_mae))
                f.write(', '.join(selected_features_name_list))
            f.close()

            # Save model
            # TODO: fix pickle file name
            filename = 'model_kmeans_clustering_best.pkl'
            pickle.dump(model, open(filename, 'wb'))
            print('saved {}'.format(filename))

        mae_results.append(res_mae)
        print('mae for {} clusters with {}: {}'.format(n_cluster, len(selected_features_name_list), res_mae))
        results_file.write('mae for {} clusters: {}\n'.format(n_cluster, res_mae))

    plt.plot(n_clusters_list, mae_results, 'b-')
    plt.grid(which='both')
    plt.xticks(list(range(10, max(n_clusters_list), 50)))
    plt.yticks(list(range(0, int(max(mae_results)), 5)))
    # plt.axis([0, max(n_clusters_list), 0, max(mae_results)])
    plt.ylabel('MAE')
    plt.xlabel('number of clusters')
    plt.show()
    plt.savefig('features/clustering_features_selection_results.png')


if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2020-04-18'
    clustering_for_features_selection(start_date, end_date)