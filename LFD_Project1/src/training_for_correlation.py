from sklearn import neural_network as NN
import pickle
from utils import DataGenerator_for_correlation_based


def main():
    start_date = '2010-01-01'
    end_date = '2020-04-06'

    training_x, training_y = DataGenerator_for_correlation_based.make_features(start_date, end_date, is_training=True)

    # TODO: set model parameters
    model = NN.MLPRegressor(hidden_layer_sizes=[70, 50, 30, 10],
                              max_iter=50000, solver='adam',
                              shuffle=True, activation='relu',learning_rate='constant',
                              early_stopping=True, batch_size='auto',
                              random_state=13, learning_rate_init=0.001,
                              n_iter_no_change=20, validation_fraction=0.2,
                              alpha=0.01, epsilon=1e-08,
                              verbose=True)
    model.fit(training_x, training_y)

    # TODO: fix pickle file name
    filename = './model/team08_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print('saved {}'.format(filename))


if __name__ == "__main__":
    main()





