import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
import matplotlib.font_manager as fm

from tqdm import tqdm

font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
mpl.rc('font', family=font_name)

mpl.rcParams['axes.unicode_minus'] = False
#%%
os.getcwd()
base_path = os.getcwd()
data_path = os.path.join(base_path, '[10] data')
cv_q = 0.9
secom_train_data = pd.read_csv(os.path.join(data_path, 'uci-secom_after_drop_cv_%d.csv' %(int(cv_q *100))))

hist_path = os.path.join(base_path, '[11] pictures')
if not os.path.exists(hist_path):
    os.mkdir(hist_path)

print('Shape of data: %s' % str(secom_train_data.shape))
print('Column names of data: %s' %(secom_train_data.columns))
print('Type of columns: %s' %(secom_train_data.dtypes))
#%%
# column 별 히스토그램 저장
df_na_rate = secom_train_data.isnull().sum() / secom_train_data.shape[0]

for col_idx in tqdm(secom_train_data.columns, desc = 'saving histogram...'):
    if not os.path.exists(os.path.join(hist_path, '%s.png' %(col_idx))):
        col_na_rate = df_na_rate.loc[col_idx]
        sns.histplot(secom_train_data.loc[:,col_idx])
        plt.title('Histogram of col_%s (NA rate: %.2f %%)' %(col_idx, col_na_rate * 100))
        plt.savefig(os.path.join(hist_path, '%s.png' %(col_idx)))
        plt.close()
#%%
# column별 NA을 각 columns별 median 값으로 대체하여 완전한 데이터셋 구축
imp_med = SimpleImputer(strategy = 'median')
imp_med.fit(secom_train_data.iloc[:,1:-1])
secom_train_data_with_imp_med = pd.DataFrame(imp_med.transform(secom_train_data.iloc[:,1:-1]))
secom_train_data_with_imp_med.columns = secom_train_data.columns[1:-1]
if not os.path.exists(os.path.join(data_path, 'uci-secom_complete_cv_%d.csv' %(int(cv_q * 100)))):
    secom_train_data_with_imp_med.to_csv(os.path.join(data_path, 'uci-secom_complete_cv_%d.csv' %(int(cv_q * 100))), index = False)
