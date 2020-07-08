import pickle
import DataGenerator
from sklearn.metrics import accuracy_score, recall_score, f1_score
import ensemble


def main():

    test_day = ['2020-01-19', '2020-02-01', '2020-02-02', '2020-02-08']
    # test_day = ['2020-02-09', '2020-02-15', '2020-02-16', '2020-02-22']
    test_x, test_y = DataGenerator.get_data(test_day, is_training=False)


    # TODO: fix pickle file name
    filename = 'team06_model.pkl'
    model, scaler = pickle.load(open(filename, 'rb'))
    print('load complete')
    print(model[0][1].get_params())

    # ================================ predict result ========================================
    test_x = scaler.transform(test_x)
    pred_y = ensemble.predict_ensemble(model, test_x)

    print('accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    print('recall: {}'.format(recall_score(test_y, pred_y)))
    print('f1-score: {}'.format(f1_score(test_y, pred_y)))



if __name__ == '__main__':
    main()