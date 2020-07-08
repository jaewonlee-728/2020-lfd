import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib import pyplot as plt
register_matplotlib_converters()

PRINT_EPOCH = 20

# ##### without constraint (Buy or Sell only one stock per day)
# def do_action(action, budget, num_stocks, stock_price):
#     # TODO : apply 10,000,000 won purchase constraints
#     # TODO: define action's operation
#     if action == "Buy" and budget >= stock_price:
#         budget -= stock_price
#         num_stocks += 1
#     elif action == "Sell" and num_stocks > 0:
#         budget += stock_price
#         num_stocks -= 1
#     else:
#         action = "Hold"
#
#     return budget, num_stocks, action

##### with constraint
def do_action(action, budget, num_stocks, stock_price):
    # TODO : apply 10,000,000 purchase constraints
    # TODO: define action's operation
    if action == "Buy" and budget >= stock_price: # Buy: buy above 10**7 won
        n_buy = min(np.ceil(10. ** 7 / stock_price), budget // stock_price)
        num_stocks += n_buy
        budget -= stock_price * n_buy
    elif action == "Sell": # Sell : sell all and buy above 10**7 won
        n_sell = num_stocks
        num_stocks = 0
        budget += stock_price * n_sell
        n_buy = min(np.ceil(10. ** 7 / stock_price), budget // stock_price)
        num_stocks += n_buy
        budget -= stock_price * n_buy
    else: # Hold : sell less than 2*10**7 won and buy above 10**7 won
        action = "Hold"
        n_sell = min(np.floor(2 * 10. ** 7 / stock_price), num_stocks)
        num_stocks -= n_sell
        budget += stock_price * n_sell
        n_buy = np.ceil(min(10. ** 7, budget) / stock_price)
        num_stocks += n_buy
        budget -= stock_price * n_buy


    return budget, num_stocks, action


def run_simulation(policy, initial_budget, initial_num_stocks, open_prices, close_prices, features):
    action_count = [0] * len(policy.actions)
    action_seq = list()

    budget = initial_budget
    num_stocks = initial_num_stocks
    stock_price = 0

    for t in range(len(open_prices)-1):
        ##### TODO: define state
        current_state = np.asmatrix(np.hstack((features[t], budget, num_stocks)))

        # calculate current portfolio value
        stock_price = float(open_prices[t])
        current_portfolio = budget + num_stocks * stock_price

        ##### select action
        action = policy.select_action(current_state, t)

        # update portfolio values based on action
        budget, num_stocks, action = do_action(action, budget, num_stocks, stock_price)
        action_seq.append(action)
        action_count[policy.actions.index(action)] += 1

        ##### TODO: define reward
        # calculate new portofolio after taking action
        stock_price = float(close_prices[t])
        new_portfolio = budget + num_stocks * stock_price

        # calculate reward from taking an action at a state
        reward = new_portfolio - current_portfolio

        ##### TODO: define state
        next_state = np.asmatrix(np.hstack((features[t+1], budget, num_stocks)))

        ##### update the policy after experiencing a new action
        policy.update_q(current_state, action, reward, next_state)

    # compute final portfolio worth
    portfolio = budget + num_stocks * stock_price
    print(
        'budget: {}, shares: {}, stock price: {} =>  portfolio: {}'.format(budget, num_stocks, stock_price, portfolio))

    return portfolio, action_count, np.asarray(action_seq)


def run_simulations(policy, budget, num_stocks, open_prices, close_prices, features, num_epoch):
    best_portfolio = 0
    final_portfolios = list()

    for epoch in range(num_epoch):
        print("-------- simulation {} --------".format(epoch + 1))
        final_portfolio, action_count, action_seq = run_simulation(policy, budget, num_stocks, open_prices, close_prices, features)
        final_portfolios.append(final_portfolio)
        print('actions : ', *zip(policy.actions, action_count))

        if (epoch + 1) % PRINT_EPOCH == 0:
            plt.figure(figsize=(40, 20))
            plt.title('Epoch {}'.format(epoch + 1))
            plt.plot(open_prices[0: len(action_seq)], 'grey')
            plt.plot(pd.DataFrame(open_prices[: len(action_seq)])[action_seq == 'Sell'], 'ro', markersize=1) # sell
            plt.plot(pd.DataFrame(open_prices[: len(action_seq)])[action_seq == 'Buy'], 'bo', markersize=1)  # buy
            plt.plot(pd.DataFrame(open_prices[: len(action_seq)])[action_seq == 'Hold'], 'go', markersize=1)  # hold
            plt.show()

        # ##### save if best portfolio value is updated
        # if best_portfolio < final_portfolio:
        #     best_portfolio = final_portfolio
        #     policy.save_model("LFD_project4-{}-e{}-step{}".format(company_list, num_epoch, epoch))

    print(final_portfolios[-1])
