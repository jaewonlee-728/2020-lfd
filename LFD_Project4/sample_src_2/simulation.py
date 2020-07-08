from pandas.plotting import register_matplotlib_converters
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
register_matplotlib_converters()

PRINT_EPOCH = 1

# ##### without constraint (Buy 1/2 of entire budget, and Sell next day)
# def do_action(action_list, action, budget, num_stocks_list, stock_price_list):
#     # TODO : apply 10,000,000 won purchase constraints
#     # TODO: define action's operation
#     for i, a in enumerate(action_list[:-1]):
#         if action == a: # Buy
#             n_buy = (budget / 2) // stock_price_list[i]
#             num_stocks_list[i] += n_buy
#             budget -= stock_price_list[i] * n_buy
#
#         else: # Not Buying
#             n_buy_list = num_stocks_list.copy()
#             num_stocks_list = [0] * (len(action_list) - 1)
#             budget += sum(stock_price_list * n_buy_list)
#
#     return budget, num_stocks_list

##### with constraint
def do_action(action_list, action, budget, num_stocks_list, stock_price_list):
    # TODO : apply 10,000,000 won purchase constraints
    # TODO: define action's operation
    for i, a in enumerate(action_list[:-1]):
        if action == a: # Buy : buy certain stock above 10**7 won, and sell next day
            budget += sum(stock_price_list * num_stocks_list)
            num_stocks_list = [0] * (len(action_list) - 1)

            n_buy = min(np.ceil(10. ** 7 / stock_price_list[i]), budget // stock_price_list[i])
            num_stocks_list[i] += n_buy
            budget -= stock_price_list[i] * n_buy

        else: # Not Buying : buy entire stock evenly above 10**7 won, and sell next day
            budget += sum(stock_price_list * num_stocks_list)
            num_stocks_list = [0] * (len(action_list) - 1)

            n_buy = min(np.ceil(10. ** 7 / sum(stock_price_list)), budget // sum(stock_price_list))
            num_stocks_list = [n_buy] * (len(action_list) - 1)
            budget -= sum(stock_price_list) * n_buy

    return budget, num_stocks_list


def run_simulation(policy, initial_budget, initial_num_stocks, open_prices, close_prices, features):
    action_count = [0] * len(policy.actions)
    action_seq = list()

    budget = initial_budget
    num_stocks_list = initial_num_stocks
    action = 'not_buying'

    for t in range(len(open_prices)-1):
        ##### TODO: define current state
        current_state = np.asmatrix(np.hstack((features[t], [budget], num_stocks_list)))

        # calculate current portfolio value
        current_portfolio = budget + sum(num_stocks_list * open_prices[t])

        ##### select action & update portfolio values
        action = policy.select_action(current_state, True)
        action_seq.append(action)
        action_count[policy.actions.index(action)] += 1

        budget, num_stocks_list = do_action(policy.actions, action, budget, num_stocks_list, open_prices[t])

        ##### TODO: define reward
        # calculate new portofolio after taking action
        new_portfolio = budget + sum(num_stocks_list * close_prices[t])
        # calculate reward from taking an action at a state
        reward = new_portfolio - current_portfolio

        ##### TODO: define next state
        next_state = np.asmatrix(np.hstack((features[t+1], [budget], num_stocks_list)))

        ##### update the policy after experiencing a new action
        policy.update_q(current_state, action, reward, next_state)

    # compute final portfolio worth
    portfolio = budget + sum(num_stocks_list * close_prices[-1])
    print('budget: {}, shares: {}, stock price: {} =>  portfolio: {}'.format(budget, num_stocks_list,
                                                                             close_prices[-1], portfolio))

    return portfolio, action_count, np.asarray(action_seq)


def run_simulations(company_list, policy, budget, num_stocks, open_prices, close_prices, features, num_epoch):
    best_portfolio = 0
    final_portfolios = list()
    for epoch in range(num_epoch):
        print("-------- simulation {} --------".format(epoch + 1))
        final_portfolio, action_count, action_seq = \
            run_simulation(policy, budget, num_stocks, open_prices, close_prices, features)
        final_portfolios.append(final_portfolio)

        print('actions : ', *zip(policy.actions, action_count), )

        if (epoch + 1) % PRINT_EPOCH == 0:
            action_seq2 = np.concatenate([['.'], action_seq[:-1]])
            for i, a in enumerate(policy.actions[:-1]):
                plt.figure(figsize=(40, 20))
                plt.title('Company {} / Epoch {}'.format(a, epoch + 1))
                plt.plot(open_prices[0: len(action_seq),i], 'grey')
                plt.plot(pd.DataFrame(open_prices[: len(action_seq), i])[action_seq2 == a], 'ro', markersize=1) # sell
                plt.plot(pd.DataFrame(open_prices[: len(action_seq), i])[action_seq == a], 'bo', markersize=1)  # buy
                plt.show()

        # ##### save if best portfolio value is updated
        # if best_portfolio < final_portfolio:
        #     best_portfolio = final_portfolio
        #     policy.save_model("LFD_project4-{}-e{}-step{}".format(company_list, num_epoch, epoch))

    print(final_portfolios[-1])