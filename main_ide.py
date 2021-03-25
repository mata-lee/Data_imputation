#%%
# main
'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd

import sys
import os
base_path = 'C:\\Users\\kt NexR\\Desktop\\mata\\work\\OJT'
if not os.path.join(base_path, '[11] code') in sys.path:
    sys.path.append(os.path.join(base_path, '[11] code'))

from data_loader import data_loader
from impute_functions import gain, Impute_med, Impute_EM
from utils import rmse_loss

from tqdm import tqdm
import itertools
import time
import datetime

#%%
def main_ide(base_path, params):
    '''Main function for UCI letter and spam datasets.

    Args:
      - data_name: letter or spam
      - miss_rate: probability of missing components
      - batch:size: batch size
      - hint_rate: hint rate
      - alpha: hyperparameter
      - iterations: iterations

    Returns:
      - imputed_data_x: (Dict) imputed data
      - rmse: Root Mean Squared Error
    '''
    base_path = base_path
    data_name = params['data_name']
    miss_rate = params['miss_rate']

    gain_parameters = {'batch_size': params['batch_size'],
                       'hint_rate': params['hint_rate'],
                       'alpha': params['alpha'],
                       'iterations': params['iterations']}

    # Load data and introduce missingness
    ori_data_x, miss_data_x, data_m = data_loader(base_path, data_name, miss_rate)

    # Impute missing data
    imputed_data_x = {}
    imputed_data_x['GAIN'] = gain(miss_data_x, gain_parameters)
    imputed_data_x['Median'] = Impute_med(miss_data_x)
    imputed_data_x['EM'] = Impute_EM(miss_data_x)

    # Report the RMSE performance
    rmse_dict = {}
    rmse_dict['GAIN'] = rmse_loss(ori_data_x, imputed_data_x['GAIN'], data_m)
    rmse_dict['Median'] = rmse_loss(ori_data_x, imputed_data_x['Median'], data_m)
    rmse_dict['EM'] = rmse_loss(ori_data_x, imputed_data_x['EM'], data_m)

    print()
    print('Parameters:')
    print(params)
    print('RMSE Performance:')
    print(rmse_dict)

    return imputed_data_x, data_m, rmse_dict
#%%
data_name_list = ['uci-secom_complete_cv_90']
miss_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
batch_size_list = [128]
hint_rate_list = [0.3, 0.5, 0.7, 0.9]
alpha_list = [100]
iterations_list = [5000]

params_list = list(itertools.product(*[data_name_list, miss_rate_list, batch_size_list, hint_rate_list, alpha_list, iterations_list]))
total_iter = len(params_list)
#%%
base_path = 'C:\\Users\\kt NexR\\Desktop\\mata\\work\\OJT'
result_path = os.path.join(base_path, '[13] result')
if not os.path.exists(result_path):
    os.mkdir(result_path)
imputed_data_path = os.path.join(result_path, 'data')
if not os.path.exists(imputed_data_path):
    os.mkdir(imputed_data_path)

result_dict = {}
total_start = time.time()

result_df = []
print('START!!')

for run_i, params_idx in enumerate(params_list):
    iter_start = time.time()
    print('%d / %d' %(run_i + 1, total_iter))
    params = {}
    params['data_name'] = params_idx[0]
    params['miss_rate'] = params_idx[1]
    params['batch_size'] = params_idx[2]
    params['hint_rate'] = params_idx[3]
    params['alpha'] = params_idx[4]
    params['iterations'] = params_idx[5]

    result_filename = '_'.join(map(str, list(params_idx)))
    if not os.path.exists(imputed_data_path):
        os.mkdir(imputed_data_path)

    imputed_data_path_iter = os.path.join(imputed_data_path, result_filename + '.xlsx')

    imputed_data_x, data_m, rmse_dict = main_ide(base_path, params)

    result_df_iter = list(params_idx) +list(rmse_dict.values())
    result_df.append(result_df_iter)

    with pd.ExcelWriter(imputed_data_path_iter) as writer:
        pd.DataFrame(imputed_data_x['GAIN']).to_excel(writer, sheet_name = 'GAIN')
        pd.DataFrame(imputed_data_x['Median']).to_excel(writer, sheet_name='Median')
        pd.DataFrame(imputed_data_x['EM']).to_excel(writer, sheet_name='EM')
        pd.DataFrame(data_m).to_excel(writer, sheet_name = 'mask')

    result_dict['_'.join(map(str, list(params.values())))] = rmse_dict

    sec_iter = time.time() - iter_start
    duration_time_iter = str(datetime.timedelta(seconds = sec_iter)).split(".")[0]
    print(duration_time_iter)

print('FINISH!!!')
sec = time.time() - total_start
duration_time = str(datetime.timedelta(seconds=sec)).split(".")[0]
print(duration_time)

result_df = pd.DataFrame(result_df)
result_df.columns = ['data_name', 'miss_rate', 'batch_size', 'hint_rate', 'alpha', 'iterations', 'RMSE_GAIN', 'RMSE_median', 'RMSE_EM']

today_date = ''.join(str(datetime.date.today()).split('-'))[2:]

result_df.to_csv(os.path.join(result_path, 'experiment_%s.csv' %(today_date)))

