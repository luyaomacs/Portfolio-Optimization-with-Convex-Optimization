import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers


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


"""### Quadratic Programming"""

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


np.random.seed(seed)

# Convert numpy arrays to cvxopt matrices
Sigma_cvx = matrix(Sigma)
q = matrix(np.zeros(len(mu)))  # The linear term is 0 in portfolio optimization
G = matrix(np.vstack((-mu, -np.identity(len(mu)))))  # Combined inequalities
h = matrix(np.hstack((-r_min, np.zeros(len(mu)))))  # Combined upper limits for inequalities
A = matrix(1.0, (1, len(mu)))  # Equality constraint coefficients
b = matrix(1.0)  # Equality constraint value

# Solve the quadratic programming problem
solvers.options['show_progress'] = False  # Optionally disable solver output
solution = solvers.qp(P=Sigma_cvx, q=q, G=G, h=h, A=A, b=b)

# Extract the optimal weights
w_qp = np.array(solution['x']).flatten()

print("Optimal portfolio weights:", w_qp)

np.random.get_state()[1][0]

num_invest = 1000000

num_shares_qp = w_qp * num_invest / stock_prices_train.iloc[0,]
no_short_train_qp = stock_prices_train.dot(num_shares_qp)

# Plot train data
plt.figure(figsize=(14, 7))
for ticker in stock_prices_train.columns:
    plt.plot(stock_prices_train.index, stock_prices_train[ticker]/stock_prices_train[ticker][0]*1000000, label=ticker)

# Plot the 'no_short_train' data、
plt.plot(no_short_train_qp.index, no_short_train_qp.values, label='no_short_train_qp', color='black')

# Add titles and labels
plt.title('Stock Prices and No Short Values Over Time (on Train)')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../res/qp_no_short_train.png')
# plt.show()

num_shares_qp = w_qp * num_invest / stock_prices_test.iloc[0,]
no_short_test_qp = stock_prices_test.dot(num_shares_qp)

actual_return = np.dot(w,mu)
actual_return_qp = np.dot(w_qp,mu)
print("actual_return = "+str(actual_return))
print("actual_return_qp = "+str(actual_return_qp))

# Plot test data
plt.figure(figsize=(14, 7))
for ticker in stock_prices_test.columns:
    plt.plot(stock_prices_test.index, stock_prices_test[ticker]/stock_prices_test[ticker][0]*1000000, label=ticker)

# Plot the 'no_short' data
plt.plot(no_short_test_qp.index, no_short_test_qp.values, label='no_short_test_qp', color='black')

# Add titles and labels
plt.title('Stock Prices and No Short Values Over Time (on Test)')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../res/qp_no_short_test.png')
# plt.show()

"""### Metrics: Volatility, Sharpe Ratio"""

# calculate volatility and sharpe_ratio on train

daily_returns = stock_prices_train.pct_change().dropna() # drop all NaN value
mean_daily_returns = np.mean(daily_returns)

# Portfolio weights
w1 = w_qp
w2 = np.array([0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857]) # uniformly invest

# Calculate portfolio returns
portfolio_returns_w1 = daily_returns.dot(w1)
portfolio_returns_w2 = daily_returns.dot(w2)

# Calculate volatility (annualized)
volatility_w1 = np.sqrt(np.dot(w1.T, np.dot(daily_returns.cov()*252, w1)))
volatility_w2 = np.sqrt(np.dot(w2.T, np.dot(daily_returns.cov()*252, w2)))

# Assume a risk-free rate for Sharpe Ratio calculation
risk_free_rate = 0.05  # Placeholder, adjust based on your benchmark

# Calculate Sharpe Ratio (annualized)
sharpe_ratio_w1 = (portfolio_returns_w1.mean() * 252 - risk_free_rate) / volatility_w1
sharpe_ratio_w2 = (portfolio_returns_w2.mean() * 252 - risk_free_rate) / volatility_w2

print(f"Volatility for Portfolio 1: {volatility_w1}")
print(f"Volatility for Portfolio 2: {volatility_w2}")
print(f"Sharpe Ratio for Portfolio 1: {sharpe_ratio_w1}")
print(f"Sharpe Ratio for Portfolio 2: {sharpe_ratio_w2}")

# calculate volatility and sharpe_ratio on test

daily_returns = stock_prices_test.pct_change().dropna()

# Portfolio weights
w1 = w_qp
w2 = np.array([0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857]) # uniformly invest

# Calculate portfolio returns
portfolio_returns_w1 = daily_returns.dot(w1)
portfolio_returns_w2 = daily_returns.dot(w2)

# Calculate volatility (annualized)
volatility_w1 = np.sqrt(np.dot(w1.T, np.dot(daily_returns.cov()*252, w1)))
volatility_w2 = np.sqrt(np.dot(w2.T, np.dot(daily_returns.cov()*252, w2)))

# Assume a risk-free rate for Sharpe Ratio calculation
risk_free_rate = 0.05  # Placeholder, adjust based on your benchmark

# Calculate Sharpe Ratio (annualized)
sharpe_ratio_w1 = (portfolio_returns_w1.mean() * 252 - risk_free_rate) / volatility_w1
sharpe_ratio_w2 = (portfolio_returns_w2.mean() * 252 - risk_free_rate) / volatility_w2

print(f"Volatility for Portfolio 1: {volatility_w1}")
print(f"Volatility for Portfolio 2: {volatility_w2}")
print(f"Sharpe Ratio for Portfolio 1: {sharpe_ratio_w1}")
print(f"Sharpe Ratio for Portfolio 2: {sharpe_ratio_w2}")

# Print main results
result_qp = {"method": "qp",
          "risk": objective(w_qp),
          "return" : np.dot(w_qp,mu),
          "volatility": volatility_w1,
          "sharpe_ratio": sharpe_ratio_w1
        }

print(result_qp)

# Print parameters
print("r_min = "+str(r_min))
print("risk_free_rate = "+str(risk_free_rate))
