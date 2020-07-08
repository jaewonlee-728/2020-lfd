import numpy as np
from sklearn.utils import resample
from itertools import combinations
from sklearn.metrics import f1_score
from sklearn import svm
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


def bootsrapping_aggregation(training_x, training_y, n_samples, random_state) :
    subsample = resample(np.concatenate((training_x, training_y.values.reshape(-1,1)), axis = 1),
                         n_samples=n_samples, replace=True, stratify=training_y,
                         random_state = random_state)
    f1 = -100000
    for i in list(combinations(range(training_x.shape[1]),2)) :

        model = svm.SVC(kernel = 'linear', max_iter = 200, random_state=0)
        model.fit(subsample[:,i], subsample[:,-1])
        label = subsample[:,-1]
        pred = model.predict(subsample[:,i])

        if f1 < f1_score(label, pred) :
            f1 = f1_score(label, pred)
            final = (i, model, f1)
    return final


def ensemble_model(training_x, training_y, n_estimator, n_samples, random_state) :
    np.random.seed(random_state)
    random_number = np.random.randint(1000000, size = n_estimator)
    best_model = []
    for ran_num in random_number :
        best_model.append(bootsrapping_aggregation(training_x, training_y, n_samples, ran_num))
    return best_model


def predict_ensemble(model_list, test_x) :
    answer = np.zeros(test_x.shape[0])

    for col, model, f1 in model_list :
        pred = model.predict(test_x[:,col])
        pred[pred==0] = -1
        pred *= f1
        answer += pred
    return (answer > 0).astype(np.int)