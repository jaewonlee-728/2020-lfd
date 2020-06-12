from sklearn import neural_network as NN
import pickle
from utils import DataGenerator_select_features_RF
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def RF_mlp_training( start_date = '2010-01-01', end_date = '2020-04-06', RF_random_state = 0,
                    mlp_random_state = 0, mlp_folder = 'mlp_model', RF_folder = 'RF',
                    input_days = 30, y_scaling = False, learning_rate = 0.0001,
                    hidden_layers = (160,80,40,20)) :
    
    if not os.path.exists(RF_folder) :
        os.mkdir(RF_folder)
    if not os.path.exists(mlp_folder) :
        os.mkdir(mlp_folder)
        
    if os.path.exists('total_table.csv') :
        total_table = pd.read_csv('total_table.csv', index_col = 0)
    else :
        total_table = DataGenerator_select_features_RF.make_dataset(start_date, end_date)
        total_table.to_csv('total_table.csv')
        
    RF = RandomForestRegressor(n_estimators=200, n_jobs = 16, random_state = RF_random_state, verbose = True)
    
    modified_set, importance_series, RF_path = DataGenerator_select_features_RF.RF_feature_selection(RF, total_table,
                                                                                                     input_days)
    
    training_x, training_y, x_scaler, y_scaler = DataGenerator_select_features_RF.data_split(modified_set, input_days,
                                                                                             y_scaling,
                                                                                             is_training=True)
    
    filename = './%s/%s' % (mlp_folder, 'RF_mlp.model')
    
    model = NN.MLPRegressor(hidden_layers, learning_rate = 'constant', verbose = True,
                        learning_rate_init=learning_rate, random_state = mlp_random_state, 
                        max_iter=10)
    
    if os.path.exists(filename) :
        with open(filename,"rb") as fr:
            print('model is already trained')
    else :
        model.fit(training_x, training_y)
        with open(filename,"wb") as fw:
            pickle.dump(model, fw)
    print('saved : {}'.format(filename))


def main():
    start_date = '2010-01-01'
    end_date = '2020-04-06'

    training_x, training_y = DataGenerator_select_features_RF.make_features(start_date, end_date, is_training=True)

    # TODO: set model parameters
    model = NN.MLPRegressor()
    model.fit(training_x, training_y)

    # TODO: fix pickle file name
    filename = 'team00_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print('saved {}'.format(filename))


if __name__ == "__main__":
    RF_mlp_training()
#     main()





