from src_1.decision_ql import QLearningDecisionPolicy
import src_1.DataGenerator as DataGenerator
import src_1.simulation as simulation
import tensorflow as tf
tf.compat.v1.reset_default_graph()


if __name__ == '__main__':
    start, end = '2010-01-01', '2020-05-19'

    # TODO: Choose companies for trading
    # company_list = ['cell', 'lgchem', 'lghnh', 'samsung1', 'samsung2', 'sk', 'hmobis', 'hmotor', 'posco', 'shinhan']
    company_list = ['samsung1']

    # TODO: define action
    actions = ["Buy", "Sell", "Hold"]


    # TODO: tuning model hyperparameters
    epsilon = 0.5
    gamma = 0.5
    lr = 0.01
    num_epoch = 20
    #########################################
    open_prices, close_prices, features = DataGenerator.make_features(company_list, start, end, is_training=True)

    budget = 10. ** 8
    num_stocks = 0
    input_dim = len(features[0]) + 1 + len(company_list)

    policy = QLearningDecisionPolicy(epsilon=epsilon, gamma=gamma, lr=lr, actions=actions, input_dim=input_dim,
                                     model_dir="model")

    simulation.run_simulations(policy=policy, budget=budget, num_stocks=num_stocks,
                               open_prices=open_prices.T[0], close_prices=close_prices.T[0], features=features,
                               num_epoch=num_epoch)

    # TODO: fix checkpoint directory name
    # policy.save_model("LFD_project4_team00-e{}".format(num_epoch))
    policy.save_model("LFD_project4_team00")
