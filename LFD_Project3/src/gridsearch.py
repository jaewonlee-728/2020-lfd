from sklearn.svm import SVC
import pickle
import DataGenerator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def main():
    test_day = ['2020-01-19', '2020-02-01', '2020-02-02', '2020-02-08']
    # test_day = ['2020-02-09', '2020-02-15', '2020-02-16', '2020-02-22']
    training_x, training_y = DataGenerator.get_data(test_day, is_training=True)

    train_x, test_x, train_y, test_y = train_test_split(training_x, training_y, test_size=0.2, random_state=1, stratify=training_y)

    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x, train_y)
    test_x = scaler.fit_transform(test_x, test_y)

    print('start tuning model')
    tuned_parameters = [{'kernel': ['rbf'], 'C': [1, 10, 100], 'gamma': [1e-3, 1e-4], 'class_weight': ['balanced'], 'max_iter': [5000, 10000, 25000, 50000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100], 'class_weight': ['balanced'], 'max_iter': [5000, 10000, 25000, 50000]}]

    scores = ['precision', 'recall']

    result_txt = open('tuning_result.txt','w', encoding='utf-8')
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        result_txt.write("# Tuning hyper-parameters for %s" % score)

        clf = GridSearchCV(
            SVC(random_state=123), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(train_x, train_y)

        result_txt.write("Best parameters set found on development set:")
        result_txt.write(clf.best_params_)
        print(clf.best_params_)
        result_txt.write("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            result_txt.write("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        result_txt.write("Detailed classification report:")
        result_txt.write("The model is trained on the full development set.")
        result_txt.write("The scores are computed on the full evaluation set.")
        y_true, y_pred = test_y, clf.predict(test_x)
        result_txt.write(classification_report(y_true, y_pred))
        print(classification_report(y_true, y_pred))
        result_txt.write("\n")

    result_txt.close()


if __name__ == '__main__':
    main()

