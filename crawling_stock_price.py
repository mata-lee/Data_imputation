import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

import seaborn as sns

import os
from tqdm import tqdm

font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
mpl.rc('font', family=font_name)

mpl.rcParams['axes.unicode_minus'] = False
#%%
# AAPL
# Ticker(종목코드) 객체를 생성
apple = yf.Ticker("AAPL")

# 회사 정보
apple.info

# 최대 기간의 주가 데이터 불러오기
price_apple = apple.history(period = 'max')
first_date_apple = str(price_apple.index[0].date())
last_date_apple = str(price_apple.index[-1].date())

# 종가 기준 그래프
price_apple['Close'].plot()
plt.title('애플 종가 그래프 (%s ~ %s)' %(first_date_apple, last_date_apple))
plt.ylabel('종가($)')
plt.show()

#%%
gme = yf.Ticker("GME")
gme.info

# 최대 기간의 주가 데이터 불러오기
price_gme = gme.history(period = 'max')
first_date_gme = str(price_gme.index[0].date())
last_date_gme = str(price_gme.index[-1].date())

# 종가 기준 그래프
price_gme['Close'].plot()
plt.title('게임스탑 종가 그래프 (%s ~ %s)' %(first_date_gme, last_date_gme))
plt.ylabel('종가($)')
plt.show()
#%%
vow = yf.Ticker("VOW.DE")
vow.info
price_vow = vow.history(period = 'max')
first_date_vow = str(price_vow.index[0].date())
last_date_vow = str(price_vow.index[-1].date())

price_vow['Close'].plot()
plt.title('폭스바겐 종가 그래프 (%s ~ %s)' %(first_date_vow, last_date_vow))
plt.ylabel('종가($)')
plt.show()

#%%
base_path = os.getcwd()
data_path = os.path.join(base_path, '[10] data')

nasdaq_filename = ['NASDAQ_%d.csv' %(i+1) for i in range(6)]
for i, filename in enumerate(nasdaq_filename):
    if i == 0:
        df_nasdaq = pd.read_csv(os.path.join(data_path, filename))
    else:
        df_nasdaq = pd.concat([df_nasdaq, pd.read_csv(os.path.join(data_path, filename))])

#%%
stock_price_data_path = os.path.join(data_path, 'stock_price')
if not os.path.exists(stock_price_data_path):
    os.mkdir(stock_price_data_path)

nasdaq_symbol = list(df_nasdaq['기호'])
if 'GME' not in nasdaq_symbol:
    nasdaq_symbol.append('GME')
#%%
def summary_tickers(symbol):
    ticker = yf.Ticker(symbol)
    ticker_hist = ticker.history(period = 'max')
    if len(price) > 0:
        first_date = str(ticker_hist.index[0].date())
        last_date = str(ticker_hist.index[-1].date())
    else: first_date, last_date = 0, 0
    return ticker_hist, first_date, last_date

nasdaq_summary = []
for symbol_idx in tqdm(nasdaq_symbol):
    ticker_hist, first_date, last_date = summary_tickers(symbol_idx)
    if not first_date == 0:
        nasdaq_summary.append([symbol_idx, first_date, last_date])

df_nasdaq_summary = pd.DataFrame(nasdaq_summary, columns = ['symbol', 'first_date', 'last_date'])
df_nasdaq_summary.to_csv(os.path.join(stock_price_data_path, 'nasdaq_summary.csv'), index = False)
df_nasdaq_summary['first_year'] = df_nasdaq_summary.apply(lambda x: int(x['first_date'].split('-')[0]), axis = 1)

standard_year = 2004
df_nasdaq_summary_trunc = df_nasdaq_summary[df_nasdaq_summary['first_year'] <= standard_year]
df_nasdaq_summary_trunc.to_csv(os.path.join(stock_price_data_path, 'nasdaq_summary_trunc.csv'), index = False)

standard_date = df_nasdaq_summary_trunc['first_date'].max()
print('Standard date: %s' %(standard_date))
#%%
collect_start_date = '2021-02-25'
collect_end_date = '2021-03-04'


gme = yf.Ticker("GME")
# gme.info
# gme.history(period = 'max')

# 최대 기간의 주가 데이터 불러오기
price_gme = gme.history(start = collect_start_date,end = collect_end_date, interval = '1m')['Close']

df_nasdaq_price = price_gme
for symbol_idx in tqdm(nasdaq_symbol):
    ticker = yf.Ticker(symbol_idx)
    price_hist = ticker.history(start = collect_start_date,end = collect_end_date, interval = '1m')['Close']
    df_nasdaq_price = pd.merge(df_nasdaq_price, price_hist, left_index= True, right_index=True, how = 'left')

df_nasdaq_price = df_nasdaq_price.iloc[:,1:]
df_nasdaq_price.columns = nasdaq_symbol
df_nasdaq_price.to_csv(os.path.join(stock_price_data_path, 'NASDAQ_top300_gme_stock_price.csv'))

df_nasdaq_na = pd.concat([pd.DataFrame(df_nasdaq_price.isnull().sum()), pd.DataFrame(df_nasdaq_price.isnull().sum() / df_nasdaq_price.shape[0])], axis = 1)
df_nasdaq_na.columns = ['count', 'rate']

df_nasdaq_na['rate'].plot()
plt.title('종목별 결측값 비율')
plt.show()

np.quantile(df_nasdaq_na['rate'], q = 0.6)

cv_na_rate =0.05
nasdaq_symbol_adj = list(df_nasdaq_na[df_nasdaq_na['rate'] <= cv_na_rate].index)
#%%
df_nasdaq_price_adj = df_nasdaq_price.loc[:, nasdaq_symbol_adj]

(df_nasdaq_price.isnull().sum() / df_nasdaq_price.shape[0]).plot()
plt.show()

sns.heatmap(df_nasdaq_price_adj.isnull(), cbar = False)
plt.show()

#%%
df_nasdaq_price_adj = df_nasdaq_price_adj.fillna(method = 'ffill').fillna(method = 'bfill')
df_nasdaq_price_adj.to_csv(os.path.join(stock_price_data_path, 'NASDAQ_top173_gme_stock_price_complete.csv'))

#%%
df_nasdaq_price_adj.to_csv(os.path.join(data_path, 'NASDAQ_top173_gme_stock_price_complete.csv'), index = False)