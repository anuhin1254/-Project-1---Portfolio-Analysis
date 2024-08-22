#!/usr/bin/env python
# coding: utf-8

# In[59]:


#Create a table showing constituent (stocks) risk analysis in the equal-weight portfolio analysis as of the
#current date.
#Column 1 – Ticker
#Column 2 – Portfolio Weight (equally weighted)
#Column 3 – Annualized Volatility (using trailing 3-months)
#Column 4 – Beta against SPY (using trailing 12-months)
#Column 5 – Beta against IWM (using trailing 12-months)
#Column 6 – Beta against DIA (using trailing 12-months
#Column 7 – Average Weekly Drawdown (52-week Low minus 52-week High) / 52-week High Column 8 – Maximum Weekly Drawdown (52-week Low minus 52-week High) / 52-week High Column 9 – Total Return (using trailing 10-years)
#Column 10 – Annualized Total Return (using trailing 10-years)

import numpy as np
import pandas as pd
import yfinance as yf

# Define the tickers in the portfolio
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'NVDA', 'SPY', 'IWM', 'DIA']

# Download historical data for the stocks and benchmarks
start_date = pd.to_datetime('2014-01-01')
end_date = pd.to_datetime('2024-02-24')
data = yf.download(tickers, start=start_date, end=end_date)['Close']
benchmarks = yf.download(['SPY', 'IWM', 'DIA'], start=start_date, end=end_date)['Close']

# Reset the index of the benchmarks DataFrame to avoid key errors
benchmarks = benchmarks.reset_index()

# Calculate the weights for an equal-weighted portfolio
weights = np.ones(len(tickers)) / len(tickers)

# Calculate the annualized volatility (standard deviation) using the trailing 3-months
volatility = data.rolling(window=63).std() * np.sqrt(252)
annualized_volatility = volatility.iloc[-1]

# Calculate the beta against SPY, IWM, and DIA using the trailing 12-months
betas = []
for i, ticker in enumerate(tickers):
    benchmark = benchmarks.loc[benchmarks['Date'].apply(lambda x: x.strftime('%Y-%m-%d')) == ticker, :]
    if not benchmark.empty:
        returns = data[ticker].pct_change()
        benchmark_returns = benchmark['Close'].pct_change()
        covariance = np.cov(returns, benchmark_returns, ddof=0)
        variance = np.var(returns, ddof=0)
        beta = covariance[0, 1] / variance
        betas.append(beta)
    else:
        betas.append(np.nan)
beta_spy = next((b for b, t in zip(betas, tickers) if t == 'SPY'), np.nan)
beta_iwm = next((b for b, t in zip(betas, tickers) if t == 'IWM'), np.nan)
beta_dia = next((b for b, t in zip(betas, tickers) if t == 'DIA'), np.nan)

# Calculate the average weekly drawdown and maximum weekly drawdown
weekly_returns = data.pct_change().rolling(window=5).mean()
high_prices = data.rolling(window=52).max()
low_prices = data.rolling(window=52).min()
drawdowns = (high_prices - data) / high_prices
average_weekly_drawdown = drawdowns.rolling(window=5).mean().iloc[-1]
maximum_weekly_drawdown = drawdowns.rolling(window=52).min().iloc[-1]

# Calculate the total return and annualized total return using the trailing 10-years
total_return = (data.iloc[-1] / data.iloc[0]) - 1
annualized_total_return = (1 + total_return) ** (252/len(data)) - 1

# Create the risk analysis table
risk_analysis = pd.DataFrame({
    'Ticker': tickers,
    'Portfolio Weight': weights,
    'Annualized Volatility': annualized_volatility,
    'Max Drawdown': maximum_weekly_drawdown,
    'Beta against SPY': beta_spy,
    'Beta against IWM': beta_iwm,
    'Beta against DIA': beta_dia,
    'Average Weekly Drawdown': average_weekly_drawdown,
    'Total Return': total_return,
    'Annualized Total Return': annualized_total_return
})
# Display the risk analysis table with the requested columns and apply color
constituent_risk_analysis = risk_analysis[[
    'Ticker',
    'Portfolio Weight',
    'Annualized Volatility',
    'Beta against SPY',
    'Average Weekly Drawdown',
    'Total Return',
    'Annualized Total Return'
]]

# Apply color to the table
def apply_color(val):
    if val < 0:
        color = 'red'
    elif val > 0:
        color = 'green'
    else:
        color = 'black'
    return 'color: %s' % color

styled_risk_analysis = constituent_risk_analysis.style.applymap(apply_color, subset=['Annualized Volatility', 'Beta against SPY', 'Average Weekly Drawdown', 'Total Return', 'Annualized Total Return'])

# Display the styled risk analysis table
styled_risk_analysis


# In[67]:


#Create a table showing Portfolio Risk against the three ETFs:
#Column 1 – ETF Ticker
#Column 2 – Correlation against ETF
#Column 3 – Covariance of Portfolio against ETF
#Column 4 – Tracking Errors (using trailing 10-years)
#Column 5 – Sharpe Ratio (using current risk-free rate)
#Column 6 – Annualized V olatility (252 days) Spread (Portfolio V olatility – ETF V olatility)


# In[70]:



# Calculate the returns for each ETF
returns = historical_data['Adj Close'].pct_change().dropna()
import yfinance as yf
import pandas as pd

# Define the ETFs you want to analyze
etfs = ['SPY', 'QQQ', 'IWM']

# Download the historical data for each ETF
historical_data = yf.download(etfs, start='2004-01-01', end='2024-02-28')

# Calculate the returns for each ETF
returns = historical_data['Adj Close'].pct_change().dropna()

# Calculate the correlation matrix
correlation = returns.corr()

# Calculate the covariance matrix
covariance = returns.cov()

# Calculate the tracking error
tracking_error = (returns - returns['SPY']).abs().mean()

# Calculate the Sharpe ratio
sharpe_ratio = returns.mean() / returns.std()

# Calculate the annualized volatility
annualized_volatility = returns.std() * pd.np.sqrt(252)

# Calculate the volatility spread
volatility_spread = (returns.std() - returns['SPY'].std()).abs()

# Save the metrics in a DataFrame
table = pd.DataFrame({
    'ETF': etfs,
    'Correlation': correlation.loc['SPY', etfs],
    'Covariance': covariance.loc['SPY', etfs],
    'Tracking Error': tracking_error[etfs],
    'Sharpe Ratio': sharpe_ratio[etfs],
    'Annualized Volatility': annualized_volatility[etfs],
    'Volatility Spread': volatility_spread[etfs],
})

print(table)



# In[74]:


#Create a correlation matrix showing the correlations between the equal-weighted portfolio,
#3 ETFs, and your 7 stocks.


# In[75]:


import pandas as pd
import numpy as np

# Calculate the returns of each asset
returns = data.pct_change()

# Calculate the correlation matrix
correlation_matrix = returns.corr()



def color_correlation(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for values
    close to -1, `'color: green'` for values
    close to 1, and black otherwise.
    """
    color = 'black'
    if np.isclose(val, -1):
        color = 'red'
    elif np.isclose(val, 1):
        color = 'green'
    return 'color: %s' % color

# Apply the color function to the correlation matrix
styled_correlation_matrix = correlation_matrix.style.applymap(color_correlation)

display(styled_correlation_matrix)

#For this project I used a lot of sources to help me put together the project. I also used a lot of my notes from CE 264 Data analysis and CE315. 
#But i will list below some of the websites i used to create this project.
#-codearmo.com
#--esoftskills.com 
#--blog.quantinsti.com
#--stackoverflow.com
#--geeksforgeeks.org
#--codecademy.com
#--GitHub
