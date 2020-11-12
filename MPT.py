import mysql.connector
import mysql.connector
import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup
import requests
from numpy.core.defchararray import lower


def connect(name_, percentage_):
    connection = mysql.connector.connect(
        database="etoilero_cryptopal",
        host="108.167.137.46",
        user="etoilero_root",
        password="new123"

    )


    mySql_insert_query = """UPDATE  Prediction SET percentage =%s WHERE coinname =%s"""
    #mySql_insert_query = """INSERT INTO  Prediction (percentage, coinname) VALUES(%s,%s)"""
    val = ( percentage_,name_)
    cursor = connection.cursor()
    cursor.execute(mySql_insert_query, val)
    connection.commit()
    print(cursor.rowcount, "Record inserted")
    cursor.close()

time =[]
combined = pd.DataFrame()


#GET TOP 10 COINS
all_coins_dict = json.loads(BeautifulSoup(
        requests.get('https://min-api.cryptocompare.com/data/top/totalvolfull?limit=10&tsym=USD&api_key=2d6f1f92684971b2967387d222f203a6d2e74d296c0caa6242c2661223783d4e').content, "html.parser").prettify())
coins = []
hisdata={}
#FILTER TO 5 COINS

for i in range(5):
    coins_=str(lower(all_coins_dict['Data'][i]['CoinInfo']['FullName']))
    coinname =(coins_.replace(' ','-'))
    coins.append(coinname)
all_coins_df = pd.DataFrame(coins)
all_coins_df.columns = ['coin']

for i in coins:
    coin_=i
    if(i=='xrp'):
        i='ripple'
    elif(i=='bitcoin-sv'):
        i='bitcoin-cash-sv'
    elif(i=='binance-coin'):
        i ='binancecoin'
    else:
        i=i
    date = []
    price = []
    something_ =[]
    request_='https://api.coingecko.com/api/v3/coins/'+i+'/market_chart/range?vs_currency=usd&from=1483228800&to' \
                                                      '=1596585600 '
    dict =json.loads(BeautifulSoup(
            requests.get(
                request_).content,
            "html.parser").prettify())
    for i in range(300):
        price.append(dict['prices'][i][1])
        date.append(dict['prices'][i][0])
        x = pd.DataFrame(date)
        x.columns = ['time']
    y = pd.DataFrame(price)
    y.columns = ['price']
    x['price'] = pd.Series(y['price'])
    x['coin'] = coin_

    combined = combined.append(x, ignore_index=True)
combined['time'] = pd.to_datetime(combined['time'], unit='ms')
combined['time'] = [d.date() for d in combined['time']]
dataframe = combined.groupby(['time', 'coin'],as_index=False)[['price']].mean()
dataframe = dataframe.set_index('time')
p_portfolio = dataframe.pivot(columns='coin')
d_returns = p_portfolio.pct_change()
period_returns = d_returns.mean()*300

daily_covariance = d_returns.cov()
period_covariance = daily_covariance*300
p_returns, p_volatility, p_sharpe_ratio, coin_weights = ([] for i in range(4))

# portfolio combinations to probe
number_of_cryptoassets = len(coins)
portfolio_number = 1000000

# for each portoflio in portfolio_number, get the return,weights and risk.
for i in range(portfolio_number):
    weights = np.random.random(number_of_cryptoassets)
    weights /= np.sum(weights)
    returns = np.dot(weights, period_returns) * 100
    volatility = np.sqrt(np.dot(weights.T, np.dot(period_covariance, weights))) * 100
    p_sharpe_ratio.append(returns / volatility)
    p_returns.append(returns)
    p_volatility.append(volatility)
    coin_weights.append(weights)
   

portfolio = {'volatility': p_volatility,
             'sharpe_ratio': p_sharpe_ratio, 'returns': p_returns}


for counter, symbol in enumerate(coins):
    portfolio[symbol + '-%'] = [Weight[counter] for Weight in coin_weights]

df = pd.DataFrame(portfolio)


order_cols = ['returns', 'volatility', 'sharpe_ratio']+[coin+'-%' for coin in coins]
df = df[order_cols]

final_portfolio = ( df.loc[df['sharpe_ratio'].idxmax()])

#connect to datbase and insert
for index, val in final_portfolio.iteritems():
    print(index,val)
    connect(index,str(val))











