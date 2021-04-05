import os
import sys
base_path = 'C:\\Users\\kt NexR\\Desktop\\mata\\work\\OJT'
if not os.path.join(base_path, '[11] code') in sys.path:
    sys.path.append(os.path.join(base_path, '[11] code'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
import matplotlib.font_manager as fm

from sklearn.metrics import r2_score, mean_squared_error

from utils import *

font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
mpl.rc('font', family=font_name)

mpl.rcParams['axes.unicode_minus'] = False
#%%
parameters = {'miss_rate': 0.4,
              'batch_size': 128,
              'hint_rate': 0.5,
              'alpha': 100,
              'iterations': 5000
              }

origin_filepath = os.getcwd() + '/[10] data'
origin_filename = 'NASDAQ_top173_gme_stock_price_complete.csv'
# origin_filename = 'uci-secom_complete_cv_90.csv'

imputation_filepath = os.getcwd() + '/[13] result/data_0326'
filename = 'NASDAQ_top173_gme_stock_price_complete_%.1f_%d_%.1f_%d_%d.xlsx' %(parameters['miss_rate'], parameters['batch_size'], parameters['hint_rate'],
                                                                               parameters['alpha'], parameters['iterations'])

origin_data = pd.read_csv(os.path.join(origin_filepath, origin_filename))
imp_GAIN = pd.read_excel(os.path.join(imputation_filepath, filename), sheet_name='GAIN', index_col=0)
imp_Med = pd.read_excel(os.path.join(imputation_filepath, filename), sheet_name='Median', index_col=0)
imp_EM = pd.read_excel(os.path.join(imputation_filepath, filename), sheet_name='EM', index_col=0)
imp_mask = pd.read_excel(os.path.join(imputation_filepath, filename), sheet_name='mask', index_col=0)

imp_GAIN.columns = origin_data.columns
imp_Med.columns = origin_data.columns
imp_EM.columns = origin_data.columns
imp_mask.columns = origin_data.columns

all_values_origin = pd.melt(origin_data)['value'] - pd.melt((origin_data * imp_mask))['value']
all_values_GAIN = pd.melt(imp_GAIN)['value'] - pd.melt((imp_GAIN * imp_mask))['value']
all_values_Med = pd.melt(imp_Med)['value'] - pd.melt((imp_Med * imp_mask))['value']
all_values_EM = pd.melt(imp_EM)['value'] - pd.melt((imp_EM * imp_mask))['value']

drop_zeros_values_origin = all_values_origin[all_values_origin != 0]
drop_zeros_values_GAIN = all_values_GAIN[all_values_GAIN != 0]
drop_zeros_values_Med = all_values_Med[all_values_Med != 0]
drop_zeros_values_EM = all_values_EM[all_values_EM != 0]

drop_zeros_values_origin.plot(linewidth = 1,label = 'Original')
drop_zeros_values_GAIN.plot(linewidth = 1,label = 'GAIN')
drop_zeros_values_Med.plot(linewidth = 1,label = 'Median')
drop_zeros_values_EM.plot(linewidth = 1,label = 'EM')
plt.legend()
plt.show()
#%%
drop_zeros_values = pd.DataFrame({'GAIN': drop_zeros_values_origin - drop_zeros_values_GAIN,
                                  'Median': drop_zeros_values_origin - drop_zeros_values_Med,
                                  'EM': drop_zeros_values_origin - drop_zeros_values_EM})

sns.histplot(drop_zeros_values)
plt.show()

score_dict = {'GAIN': [r2_score(drop_zeros_values_origin, drop_zeros_values_GAIN), rmse_loss(origin_data.T.to_numpy(), imp_GAIN.T.to_numpy(), imp_mask.T.to_numpy())],
              'Median': [r2_score(drop_zeros_values_origin, drop_zeros_values_Med), rmse_loss(origin_data.T.to_numpy(), imp_Med.T.to_numpy(), imp_mask.T.to_numpy())],
              'EM': [r2_score(drop_zeros_values_origin, drop_zeros_values_EM), rmse_loss(origin_data.T.to_numpy(), imp_EM.T.to_numpy(), imp_mask.T.to_numpy())]}
summuary_result = drop_zeros_values.describe()

summuary_result = pd.concat([summuary_result, pd.DataFrame(score_dict, index = ['R^2', 'norm_rmse'])])
#%%
print(f'RMSE loss: {summuary_result.loc["norm_rmse"]}')
#%%
# GME stock price
# Miss_rate for GME
print(f'Miss_rate for GME: {(1- imp_mask).sum().sum() / (imp_mask.shape[0] * imp_mask.shape[1])}')

((1- imp_mask).sum() / imp_mask.shape[0]).plot(kind = 'bar')
plt.axhline(y = 0.40, color = 'red')
plt.title(f'종목별 결측값 비율(miss_rate: {parameters["miss_rate"]})')
plt.ylim(0.35, 0.45)
plt.show()

df_GME_stock_price = pd.concat([origin_data.iloc[:,-1], imp_GAIN.iloc[:,-1], imp_Med.iloc[:,-1], imp_EM.iloc[:,-1], imp_mask.iloc[:,-1]], axis = 1)
df_GME_stock_price.columns = ['Original', 'GAIN', 'Median', 'EM', 'Mask']

# Check median value if mask == 1
# np.median(df_GME_stock_price['Median'].loc[df_GME_stock_price['Mask'] == 1])

# Check median value if mask == 1
# df_GME_stock_price['Original'].loc[df_GME_stock_price['Mask'] == 0].plot()
# df_GME_stock_price['GAIN'].loc[df_GME_stock_price['Mask'] == 0].plot()
# df_GME_stock_price['Median'].loc[df_GME_stock_price['Mask'] == 0].plot()
# df_GME_stock_price['EM'].loc[df_GME_stock_price['Mask'] == 0].plot()
# plt.show()

df_GME_stock_price.loc[df_GME_stock_price['Mask'] == 0,['Original', 'GAIN', 'Median']].plot()
plt.show()
#%%
# Check max std: GME
print(f'Max std column: {origin_data.columns[np.argmax(origin_data.std())]}')
print(f'Min std column: {origin_data.columns[np.argmin(origin_data.std())]}')

origin_data.std().plot()
plt.title('Std of Stocks')
plt.show()
#%%
