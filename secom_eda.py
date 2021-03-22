import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
import matplotlib.font_manager as fm

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

(secom_train_data.isnull().sum() / secom_train_data.shape[0]).sort_values(ascending = False).plot(kind = 'bar')
plt.title('컬럼별 NA 비율')
plt.show()
#%%



(secom_train_data.isnull().sum() == 0).sum() / secom_train_data.shape[1]
secom_train_data.isnull().sum().describe()

sns.boxplot(secom_train_data.isnull().sum()/ secom_train_data.shape[0])
plt.show()

np.quantile(secom_train_data.isnull().sum(axis = 1) / secom_train_data.shape[1], q = 0.90)
np.quantile(secom_train_data.isnull().sum(axis = 1) / secom_train_data.shape[1], q = 0.75)

sns.boxplot(secom_train_data.isnull().sum(axis = 1)/ secom_train_data.shape[1])
plt.show()

sns.boxplot(secom_train_data.isnull().sum(axis = 1))
plt.show()

#%%
np.quantile(secom_train_data.isnull().sum(axis = 0) / secom_train_data.shape[0], q = 0.90)
np.quantile(secom_train_data.isnull().sum(axis = 0) / secom_train_data.shape[0], q = 0.75)

sns.boxplot(secom_train_data.isnull().sum(axis = 0)/ secom_train_data.shape[0])
plt.show()

sns.boxplot(secom_train_data.isnull().sum(axis = 0))
plt.show()


#%%
cut_rank = 52
secom_train_data

secom_train_data.shape[0]
(secom_train_data.isnull().sum(axis = 0) * 100 / secom_train_data.shape[0]).sort_values(ascending=False).iloc[cut_rank]



