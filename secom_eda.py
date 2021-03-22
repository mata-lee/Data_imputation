import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
import matplotlib.font_manager as fm

font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
mpl.rc('font', family=font_name)

mpl.rcParams['axes.unicode_minus'] = False
#%%
os.getcwd()
base_path = os.getcwd()
data_path = os.path.join(base_path, '[10] data')
secom_train_data = pd.read_csv(os.path.join(data_path, 'uci-secom.csv'))

print('Shape of data: %s' % str(secom_train_data.shape))
print('Column names of data: %s' %(secom_train_data.columns))
print('Type of columns: %s' %(secom_train_data.dtypes))
#%%
secom_train_data.isna().sum(axis = 0).sum() / (secom_train_data.shape[0] * secom_train_data.shape[1])
secom_train_data.isna().sum(axis = 1) / secom_train_data.shape[0]

# NA가 다수 존재하는 columns이 있음.
sns.heatmap(secom_train_data.T.isnull(), cbar = False)
plt.xlabel('rows')
plt.ylabel('columns')
plt.show()
#%%
# 각 columns별 NA 수 파악
secom_train_data.isnull().sum().sort_values(ascending = False)

# 임계값 설정하여 NA가 일정 비율 이상인 columns은 삭제
cv_q = 0.9 # NA 비율 분포 중 90% quantile 이상(0.0308, 약 3.08%)이 NA인 columns은 삭
print('NA 비율 quantile 임계값: %.2f' %(cv_q))
cv_rate_value = np.quantile((secom_train_data.isnull().sum() / secom_train_data.shape[0]), q = cv_q)
print('NA 비율 rate 임계값: %.2f %%' %(cv_rate_value * 100))

drop_col_rate = sum(((secom_train_data.isnull().sum() / secom_train_data.shape[0]) > np.quantile((secom_train_data.isnull().sum() / secom_train_data.shape[0]), q = cv_q))) / secom_train_data.shape[1]
drop_col_num = sum(((secom_train_data.isnull().sum() / secom_train_data.shape[0]) > np.quantile((secom_train_data.isnull().sum() / secom_train_data.shape[0]), q = cv_q)))
print('NA 비율이 높아 삭제되는 columns 비율(개수): %.2f %% (%d)' %(drop_col_rate * 100, drop_col_num))

(secom_train_data.isnull().sum() / secom_train_data.shape[0]).sort_values(ascending = False).plot(kind = 'bar')
plt.hist(range(0, drop_col_num), color = 'red', alpha = 0.3, bins = 1)
plt.axhline(y = np.quantile((secom_train_data.isnull().sum() / secom_train_data.shape[0]), q = cv_q), c = 'red')
plt.text(x = 70.0, y = np.quantile((secom_train_data.isnull().sum() / secom_train_data.shape[0]), q = 0.9) + 0.03, s = '90%% quantile: %.2f' %(np.quantile((secom_train_data.isnull().sum() / secom_train_data.shape[0]), q = 0.9)),
         fontdict = {'color': 'red', 'weight': 'bold', 'size': 14})
plt.ylim(0,1)
plt.title('컬럼별 NA 비율')
plt.show()
#%%
drop_condition = (secom_train_data.isnull().sum() / secom_train_data.shape[0]) > cv_rate_value
drop_cols = secom_train_data.columns[drop_condition]

secom_train_data_after_drop = secom_train_data.drop(drop_cols, axis = 'columns')
print('NA 비율 높은 columns 제거 후의 데이터 형태:', secom_train_data_after_drop.shape)
#%%
# NA 결측이 높은 column 삭제 후 데이터 저장
if not os.path.exists(os.path.join(data_path, 'uci-secom_after_drop_cv_%d.csv' %(int(cv_q * 100)))):
    secom_train_data_after_drop.to_csv(os.path.join(data_path, 'uci-secom_after_drop_cv_%d.csv' %(int(cv_q * 100))), index = False)
