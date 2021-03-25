import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

import os
from tqdm import tqdm

# For Window
font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
mpl.rc('font', family=font_name)

# For mac
mpl.rcParams['font.family'] = "AppleGothic"


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

nasdaq_filename = ['NASDAQ_%d.csv' %(i+1) for i in range(3)]
for i, filename in enumerate(nasdaq_filename):
    if i == 0:
        df_nasdaq = pd.read_csv(os.path.join(data_path, filename))
    else:
        df_nasdaq = pd.concat([df_nasdaq, pd.read_csv(os.path.join(data_path, filename))])

# 중복되는 회사 제외(ex. 알파벳 C, 알파벳 A)
df_nasdaq['종목2'] = df_nasdaq['종목'].apply(lambda x: x.split()[0])
nasdaq_symbol = df_nasdaq.drop_duplicates('종목2')['기호']

#%%
# 