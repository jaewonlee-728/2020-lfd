import numpy as np
import src_2.DataGenerator as DataGenerator
from src_2.decision_ql import QLearningDecisionPolicy
from src_2.simulation import do_action


def run(policy, initial_budget, initial_num_stocks, open_prices, close_prices, features):

    budget = initial_budget
    num_stocks_list = initial_num_stocks

    for i in range(len(open_prices)):
        current_state = np.asmatrix(np.hstack((features[i], [budget], num_stocks_list)))
        action = policy.select_action(current_state, is_training=False)
        budget, num_stocks_list = do_action(policy.actions, action, budget, num_stocks_list, open_prices[i])
        print('Day {}'.format(i+1))
        print('action {} / budget {} / shares {}'.format(action, budget, num_stocks_list))
        print('portfolio with  open price : {}'.format(budget + sum(num_stocks_list * open_prices[i])))
        print('portfolio with close price : {}\n'.format(budget + sum(num_stocks_list * close_prices[i])))

    portfolio = budget + sum(num_stocks_list * close_prices[-1])

    print('Finally, you have')
    print('budget: %.2f won' % budget)
    print('Share : {}'.format(num_stocks_list))
    print('Share value : {} won'.format(close_prices[-1]))
    print()

    return portfolio


if __name__ == '__main__':
    start, end = '2010-01-01', '2020-05-19'
    company_list = ['lgchem', 'samsung1']
    actions = company_list + ['not_buying']
    #########################################
    open_prices, close_prices, features = DataGenerator.make_features(company_list, start, end, is_training=False)
    budget = 10. ** 8
    num_stocks = [0] * len(company_list)
    input_dim = len(features[0]) + 1 + len(company_list)

    # TODO: fix checkpoint directory name
    # policy = QLearningDecisionPolicy(0, 1, 0, actions, input_dim, "LFD_project4_team00-e20")
    policy = QLearningDecisionPolicy(0, 1, 0, actions, input_dim, "LFD_project4_team00")
    final_portfolio = run(policy, budget, num_stocks, open_prices, close_prices, features)

    print("Final portfolio: %.2f won" % final_portfolio)

