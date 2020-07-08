from sklearn import svm
import pickle
import DataGenerator
import ensemble
from sklearn.preprocessing import StandardScaler

def main():
    test_day = ['2020-01-19', '2020-02-01', '2020-02-02', '2020-02-08']
    # test_day = ['2020-02-09', '2020-02-15', '2020-02-16', '2020-02-22']
    training_x, training_y = DataGenerator.get_data(test_day, is_training=True)

    # ================================ train SVM model=========================================
    # TODO: set parameters
    print('start training model')

    scaler = StandardScaler()
    scaler.fit(training_x)
    training_x = scaler.transform(training_x)

    model = ensemble.ensemble_model(training_x, training_y, 10, 2300, 4)

    print('completed training model')

    # TODO: fix pickle file name
    filename = 'team06_model.pkl'

    pickle.dump((model, scaler), open(filename, 'wb'))
    print('save complete')


if __name__ == '__main__':
    main()

