# Portfolio-Optimization-with-Convex-Optimization

Implemented two methods (quadratic programming (QP), sequential least squares programming (SLSQP)) for solving portfolio optimization problem. Use `minimize(w^T @ Sigma @ w)` as objective function, where `w` is the weights for asset assignment and `Sigma` is the covariance matrix of asset returns.

## Get started

### Install dependency packages

Install `yfinance` package for downloading stock data.
```
pip install yfinance
```
Install `cvxopt` packages for solving quadratic programming problem.
```
pip install cvxopt
```

### Run code

You can run the code under `src` directory using following commands:
```
python slsqp_solver.py
```
```
python qp_solver.py
```

Result plots would be saved under `res` directory.

## References
https://kyle-stahl.com/stock-portfolio-optimization