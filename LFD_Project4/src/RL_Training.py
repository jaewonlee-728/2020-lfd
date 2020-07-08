from decision_ql import QLearningDecisionPolicy
import DataGenerator as DataGenerator
import simulation as simulation
import tensorflow as tf
tf.compat.v1.reset_default_graph()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    start, end = '2010-01-01', '2020-05-19'

    # TODO: Choose companies for trading
    # company_list = ['cell', 'hmotor', 'naver', 'lgchem', 'lghnh', 'samsung1', 'sdi', 'sk', 'kospi']
    company_list = ['cell', 'sdi', 'sk', 'lgchem']

    # TODO: define action
    # actions = company_list + ['not_buying']
    action_to_company = [[li+'_full', li+'_half'] for li in company_list]
    action_to_company = sum(action_to_company, [])
    actions = action_to_company + ['full_buying', 'half_buying', 'minimum_buying']

    #########################################
    # TODO: tuning model hyperparameters
    epsilon = 0.2 #0.5
    gamma = 0.5
    lr = 0.001
    num_epoch = 20

    #########################################
    open_prices, close_prices, features = DataGenerator.make_features(company_list, start, end, is_training=True)
    budget = 10. ** 8
    num_stocks = [0] * len(company_list)

    input_dim = len(features[0]) + 1 + len(company_list)
    policy = QLearningDecisionPolicy(epsilon=epsilon, gamma=gamma, lr=lr, actions=actions, input_dim=input_dim,
                                     model_dir="model")

    simulation.run_simulations(company_list=company_list, policy=policy, budget=budget, num_stocks=num_stocks,
                               open_prices=open_prices, close_prices=close_prices, features=features,
                               num_epoch=num_epoch)

    # TODO: fix checkpoint directory name
    # policy.save_model("LFD_project4_team00-e{}".format(num_epoch))
    policy.save_model("LFD_project4_team05")
