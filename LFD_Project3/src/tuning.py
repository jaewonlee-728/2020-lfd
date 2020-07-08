from sklearn import svm
import pickle
import DataGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import resample
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


def main():
    test_day = ['2020-01-19', '2020-02-01', '2020-02-02', '2020-02-08']
    # test_day = ['2020-02-09', '2020-02-15', '2020-02-16', '2020-02-22']
    training_x, training_y = DataGenerator.get_data(test_day, is_training=True)
    test_x, test_y = DataGenerator.get_data(test_day, is_training=False)

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
    test_x = scaler.fit_transform(test_x, test_y)

    # ================================ train SVM model=========================================
    # TODO: set parameters
    print('start training model')
    # model = svm.SVC(C=1, kernel='linear', random_state=0, class_weight={0: 5, 1: 5})

    kernel_list = ['rbf', 'linear']
    C_list = [1, 100, 1000]

    for kernel in kernel_list:
        for C in C_list:
            model = svm.SVC(C=C, kernel=kernel, random_state=123)
            model.fit(training_x, training_y)

            pred_y = model.predict(test_x)

            print("kernel, C: {},{}".format(kernel, C))
            print()
            print('precision: {}'.format(precision_score(test_y, pred_y)))
            print('accuracy: {}'.format(accuracy_score(test_y, pred_y)))
            print('recall: {}'.format(recall_score(test_y, pred_y)))
            print('f1-score: {}'.format(f1_score(test_y, pred_y)))


if __name__ == '__main__':
    main()

