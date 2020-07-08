from pandas.plotting import register_matplotlib_converters
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
register_matplotlib_converters()

PRINT_EPOCH = 20




def do_action(action_list, action, budget, num_stocks_list, stock_price_list):
    # TODO : apply 10,000,000 won purchase constraints
    # TODO: define action's operation

    # full
    for i, a in enumerate(action_list[0:-3:2]):
        if action == a: # Buy : buy certain stock above 10**7 won, and sell next day
            budget += sum(stock_price_list * num_stocks_list)
            num_stocks_list = [0] * int((len(action_list) - 3)/2)
            n_buy = budget// stock_price_list[i]
            num_stocks_list[i] += n_buy
            budget -= stock_price_list[i] * n_buy

    # half
    for i, a in enumerate(action_list[1:-3:2]):
        if action == a: # Buy : buy certain stock above 10**7 won, and sell next day
            budget += sum(stock_price_list * num_stocks_list)
            num_stocks_list = [0] * int((len(action_list) - 3)/2)

            n_buy = min(budget/2 // stock_price_list[i], budget // stock_price_list[i])
            num_stocks_list[i] += n_buy
            budget -= stock_price_list[i] * n_buy


    if action == "full_buying":
        budget += sum(stock_price_list * num_stocks_list)
        # num_stocks_list = [0] * len(num_stocks_list)
        n_buy = budget/len(num_stocks_list) // stock_price_list
        num_stocks_list = n_buy
        budget -= sum(stock_price_list * n_buy)

    elif action == "half_buying":
        budget += sum(stock_price_list * num_stocks_list)
        # num_stocks_list = [0] * len(num_stocks_list)
        n_buy = budget/2/len(num_stocks_list) // stock_price_list
        num_stocks_list = n_buy
        budget -= sum(stock_price_list * n_buy)

    elif action == "minimum_buying":

        budget += sum(stock_price_list * num_stocks_list)
        # num_stocks_list = [0] * len(num_stocks_list)
        n_buy = np.ceil(10. ** 7/len(num_stocks_list) // stock_price_list)
        num_stocks_list = n_buy
        budget -= sum(stock_price_list * n_buy)

    return budget, num_stocks_list




def run_simulation(policy, initial_budget, initial_num_stocks, open_prices, close_prices, features):
    action_count = [0] * len(policy.actions)
    action_seq = list()

    budget = initial_budget
    num_stocks_list = initial_num_stocks

    for t in range(len(open_prices)-1):
        ##### TODO: define current state
        current_state = np.asmatrix(np.hstack((features[t], [budget], num_stocks_list)))
        # calculate current portfolio value
        current_portfolio = budget + sum(num_stocks_list * open_prices[t])

        ##### select action & update portfolio values
        action = policy.select_action(current_state, True)
        action_seq.append(action)
        # for act in action:
        action_count[policy.actions.index(action)] += 1

        budget, num_stocks_list = do_action(policy.actions, action, budget, num_stocks_list, open_prices[t])

        ##### TODO: define reward
        # calculate new portfolio after taking action

        stock_price = sum(num_stocks_list * close_prices[t])
        new_portfolio = budget + stock_price
        # calculate reward from taking an action at a state

        if budget < stock_price :
            reward = (new_portfolio - current_portfolio)  * 1
        if budget > stock_price :
            reward = (new_portfolio - current_portfolio) * 20
        if np.abs(budget - stock_price) < 10000000 :
            reward = (new_portfolio - current_portfolio)  *  10

        ##### TODO: define next state
        next_state = np.asmatrix(np.hstack((features[t+1], [budget], num_stocks_list)))

        ##### update the policy after experiencing a new action
        policy.update_q(current_state, action, reward, next_state)

    # compute final portfolio worth
    portfolio = budget + sum(num_stocks_list * close_prices[-1])
    if portfolio < 100000000 :
        print('budget: {}, shares: {}, stock price: {} =>  portfolio: {}_@@'.format(budget, num_stocks_list,
                                                                                 close_prices[-1], portfolio))
    else :
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

        # if (epoch + 1) % PRINT_EPOCH == 0:
        #     action_seq2 = np.concatenate([['.'], action_seq[:-1]])
        #     for i, a in enumerate(policy.actions[:-1]):
        #         plt.figure(figsize=(40, 20))
        #         plt.title('Company {} / Epoch {}'.format(a, epoch + 1))
        #         plt.plot(open_prices[0: len(action_seq),i], 'grey')
        #         plt.plot(pd.DataFrame(open_prices[: len(action_seq), i])[action_seq2 == a], 'ro', markersize=1) # sell
        #         plt.plot(pd.DataFrame(open_prices[: len(action_seq), i])[action_seq == a], 'bo', markersize=1)  # buy
        #         plt.show()

        # ##### save if best portfolio value is updated
        # if best_portfolio < final_portfolio:
        #     best_portfolio = final_portfolio
        #     policy.save_model("LFD_project4-{}-e{}-step{}".format(company_list, num_epoch, epoch))

    print(final_portfolios[-1])