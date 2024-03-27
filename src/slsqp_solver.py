import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# download stock data
stock_prices_train = yf.download(tickers = "AMZN NVDA NFLX MSFT ORCL BAC JPM",
                          start = "2020-01-01",
                          end = "2023-01-01",
                          progress=False)["Adj Close"]

# # plot train data
# plt.figure(figsize=(14, 7))

# for ticker in stock_prices_train.columns:
#     plt.plot(stock_prices_train[ticker], label=ticker)

# plt.title('Stock Prices 2020-2023 (Train)')
# plt.xlabel('Date')
# plt.ylabel('Adjusted Closing Price')
# plt.legend()
# plt.grid(True)
# plt.show()

stock_prices_test = yf.download(tickers = "AMZN NVDA NFLX MSFT ORCL BAC JPM",
                          start = "2023-01-01",
                          end = "2024-01-01",
                          progress=False)["Adj Close"]

# # plot test data
# plt.figure(figsize=(14, 7))
# for ticker in stock_prices_test.columns:
#     plt.plot(stock_prices_test[ticker], label=ticker)

# plt.title('Stock Prices 2023-2024 (Test)')
# plt.xlabel('Date')
# plt.ylabel('Adjusted Closing Price')
# plt.legend()
# plt.grid(True)
# plt.show()

returns = stock_prices_train.pct_change() # calculate the daily return of each stock price
returns = returns.iloc[1:]
returns = returns + 1

"""### Sequential Least Squares Programming"""

# Mean return for each stock (in array)
mu = np.array(np.mean(returns, axis=0))

# Covariance matrix between stocks (in array)
Sigma = np.array(returns.cov())

# Vector of 1's with len(mu)
e = np.ones(len(mu))

# Setting the expected average daily return of the portfolio
r_min = 1+(0.10/252) # 10% rate annually per day

def objective(w):
    return np.matmul(np.matmul(w,Sigma),w) # w^T S w

# Set initial weight values
seed = 3407
np.random.seed(seed)
w = np.random.random(len(mu))

# Define Constraints
const = ({'type' : 'ineq' , 'fun' : lambda w: np.dot(w,mu) - r_min}, # u^T w - r_min >= 0
         {'type' : 'eq' , 'fun' : lambda w: np.dot(w,e) - 1})    # sum(w) - 1 = 0

# Create Bounds
# Creates a tuple of tuples to pass to minimize
# to ensure all weights are betwen [0, inf]
# no-short investment, therefore we have w >= 0
non_neg = []
for i in range(len(mu)):
    non_neg.append((0,None))
non_neg = tuple(non_neg)

# Run optimization with SLSQP solver
solution = minimize(fun=objective, x0=w, method='SLSQP', constraints=const, bounds=non_neg)

w = solution.x.round(6)
# print(w)
# print(w.sum())
# print(list(returns.columns[w > 0.0]))

np.random.get_state()[1][0] # get the random seed

# invest on train data according to weight
num_invest = 1000000
num_shares = w * num_invest / stock_prices_train.iloc[0,]
no_short_train = stock_prices_train.dot(num_shares)

# Plot train data
plt.figure(figsize=(14, 7))
for ticker in stock_prices_train.columns:
    plt.plot(stock_prices_train.index, stock_prices_train[ticker]/stock_prices_train[ticker][0]*num_invest, label=ticker)

# Plot the 'no_short_train' data
plt.plot(no_short_train.index, no_short_train.values, label='no_short_train', color='black')

# Add titles and labels
plt.title('Stock Prices and No Short Values Over Time (on Train)')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../res/slsqp_no_short_train.png')
# plt.show()

# invest on test data according to weight

num_shares = w * num_invest / stock_prices_test.iloc[0,]
no_short_test = stock_prices_test.dot(num_shares)
no_short_test

# Plot test data
plt.figure(figsize=(14, 7))
for ticker in stock_prices_test.columns:
    plt.plot(stock_prices_test.index, stock_prices_test[ticker]/stock_prices_test[ticker][0]*1000000, label=ticker)

# Plot the 'no_short_test' data
plt.plot(no_short_test.index, no_short_test.values, label='no_short_test', color='black')

# Add titles and labels
plt.title('Stock Prices and No Short Values Over Time (on Test)')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../res/slsqp_no_short_test.png')
# plt.show()


"""### Metrics: Volatility, Sharpe Ratio"""

# calculate volatility and sharpe_ratio on train

daily_returns = stock_prices_train.pct_change().dropna() # drop all NaN value
mean_daily_returns = np.mean(daily_returns)

# Portfolio weights
w0 = w
w2 = np.array([0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857]) # uniformly invest

# Calculate portfolio returns
portfolio_returns_w0 = daily_returns.dot(w0)
portfolio_returns_w2 = daily_returns.dot(w2)

# Calculate volatility (annualized)
volatility_w0 = np.sqrt(np.dot(w0.T, np.dot(daily_returns.cov()*252, w0)))
volatility_w2 = np.sqrt(np.dot(w2.T, np.dot(daily_returns.cov()*252, w2)))

# Assume a risk-free rate for Sharpe Ratio calculation
risk_free_rate = 0.05  # Placeholder, adjust based on your benchmark

# Calculate Sharpe Ratio (annualized)
sharpe_ratio_w0 = (portfolio_returns_w0.mean() * 252 - risk_free_rate) / volatility_w0
sharpe_ratio_w2 = (portfolio_returns_w2.mean() * 252 - risk_free_rate) / volatility_w2

print(f"Volatility for Portfolio 0: {volatility_w0}")
print(f"Volatility for Portfolio 2: {volatility_w2}")
print(f"Sharpe Ratio for Portfolio 0: {sharpe_ratio_w0}")
print(f"Sharpe Ratio for Portfolio 2: {sharpe_ratio_w2}")

# calculate volatility and sharpe_ratio on test

daily_returns = stock_prices_test.pct_change().dropna()

# Portfolio weights
w0 = w
w2 = np.array([0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857]) # uniformly invest

# Calculate portfolio returns
portfolio_returns_w0 = daily_returns.dot(w0)
portfolio_returns_w2 = daily_returns.dot(w2)

# Calculate volatility (annualized)
volatility_w0 = np.sqrt(np.dot(w0.T, np.dot(daily_returns.cov()*252, w0)))
volatility_w2 = np.sqrt(np.dot(w2.T, np.dot(daily_returns.cov()*252, w2)))

# Assume a risk-free rate for Sharpe Ratio calculation
risk_free_rate = 0.05  # Placeholder, adjust based on your benchmark

# Calculate Sharpe Ratio (annualized)
sharpe_ratio_w0 = (portfolio_returns_w0.mean() * 252 - risk_free_rate) / volatility_w0
sharpe_ratio_w2 = (portfolio_returns_w2.mean() * 252 - risk_free_rate) / volatility_w2

print(f"Volatility for Portfolio 0: {volatility_w0}")
print(f"Volatility for Portfolio 2: {volatility_w2}")
print(f"Sharpe Ratio for Portfolio 0: {sharpe_ratio_w0}")
print(f"Sharpe Ratio for Portfolio 2: {sharpe_ratio_w2}")

# Print main results
result_slsqp = {"method": "slsqp",
          "risk": objective(w),
          "return" : np.dot(w,mu),
          "volatility": volatility_w0,
          "sharpe_ratio": sharpe_ratio_w0
        }

print(result_slsqp)

# Print parameters
print("r_min = "+str(r_min))
print("risk_free_rate = "+str(risk_free_rate))
