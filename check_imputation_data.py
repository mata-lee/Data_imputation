import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_squared_error

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

origin_data.columns = imp_GAIN.columns
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
# #%%
#
# drop_zeros_values_origin.plot(linewidth = 1,label = 'Original')
# drop_zeros_values_GAIN.plot(linewidth = 1,label = 'GAIN')
# drop_zeros_values_Med.plot(linewidth = 1,label = 'Median')
# drop_zeros_values_EM.plot(linewidth = 1,label = 'EM')
# plt.legend()
# plt.ylim(0,500)
# plt.show()
#%%
drop_zeros_values = pd.DataFrame({'GAIN': drop_zeros_values_origin - drop_zeros_values_GAIN,
                                  'Median': drop_zeros_values_origin - drop_zeros_values_Med,
                                  'EM': drop_zeros_values_origin - drop_zeros_values_EM})

sns.histplot(drop_zeros_values)
plt.show()

r2_score_dict = {'GAIN': [r2_score(drop_zeros_values_origin, drop_zeros_values_GAIN), mean_squared_error(drop_zeros_values_origin, drop_zeros_values_GAIN) ** 0.5],
                 'Median': [r2_score(drop_zeros_values_origin, drop_zeros_values_Med), mean_squared_error(drop_zeros_values_origin, drop_zeros_values_Med) ** 0.5],
                 'EM': [r2_score(drop_zeros_values_origin, drop_zeros_values_EM), mean_squared_error(drop_zeros_values_origin, drop_zeros_values_EM) ** 0.5]}
summuary_result = drop_zeros_values.describe()
summuary_result = pd.concat([summuary_result, pd.DataFrame(r2_score_dict, index = ['R^2', 'rmse'])])
#%%
# GME stock price
imp_mask.sum().sum() / (imp_mask.shape[0] * imp_mask.shape[1])
imp_mask.iloc[:,-1].sum() / imp_mask.shape[0]

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
np.argmax(origin_data.std())
np.argmin(origin_data.std())

origin_data.std().plot()
plt.title('Std of Stocks')
plt.show()