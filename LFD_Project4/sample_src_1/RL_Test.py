import numpy as np
import src_1.DataGenerator as DataGenerator
from src_1.decision_ql import QLearningDecisionPolicy
from src_1.simulation import do_action


def run(policy, initial_budget, initial_num_stocks, open_prices, close_prices, features):

    budget = initial_budget
    num_stocks = initial_num_stocks

    for i in range(len(open_prices)):
        current_state = np.asmatrix(np.hstack((features[i], budget, num_stocks)))
        action = policy.select_action(current_state, is_training=False)
        stock_price = float(open_prices[i])
        budget, num_stocks, action = do_action(action, budget, num_stocks, stock_price)

        print('Day {}'.format(i+1))
        print('action {} / budget {} / shares {}'.format(action, budget, num_stocks))
        print('portfolio with  open price : {}'.format(budget + num_stocks * open_prices[i]))
        print('portfolio with close price : {}\n'.format(budget + num_stocks * close_prices[i]))

    portfolio = budget + num_stocks * close_prices[-1]

    print('Finally, you have')
    print('budget: %.2f won' % budget)
    print('Shares: %i' % num_stocks)
    print('Share value: %.2f won' % close_prices[-1])
    print()

    return portfolio


if __name__ == '__main__':
    start, end = '2010-01-01', '2020-05-19'
    company_list = ['samsung1']
    actions = ["Buy", "Sell", "Hold"]
    #########################################
    open_prices, close_prices, features = DataGenerator.make_features(company_list, start, end, is_training=False)

    budget = 10. ** 8
    num_stocks = 0
    input_dim = len(features[0]) + 1 + len(company_list)

    # TODO: fix checkpoint directory name
    # policy = QLearningDecisionPolicy(0, 1, 0, actions, input_dim, "LFD_project4_team00-e20")
    policy = QLearningDecisionPolicy(0, 1, 0, actions, input_dim, "LFD_project4_team00")
    final_portfolio = run(policy, budget, num_stocks, open_prices.T[0], close_prices.T[0], features)

    print("Final portfolio: %.2f won" % final_portfolio)

