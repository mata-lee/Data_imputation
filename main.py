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

import sys
import os
base_path = 'C:\\Users\\kt NexR\\Desktop\\mata\\work\\OJT'
if not os.path.join(base_path, '[11] code') in sys.path:
    sys.path.append(os.path.join(base_path, '[11] code'))

from data_loader import data_loader
from impute_functions import gain, Impute_med, Impute_EM
from utils import rmse_loss

"""
data_name = os.path.join(os.getcwd(), '[10] data/uci-secom_complete_cv_90.csv')
data_name = 'uci-secom_complete_cv_90'
miss_rate = 0.3
gain_parameters = {'batch_size': 128, 'hint_rate': 0.5, 'alpha': 100, 'iterations': 5000}
"""
def main(args):
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

    data_name = args.data_name
    miss_rate = args.miss_rate

    gain_parameters = {'batch_size': args.batch_size,
                       'hint_rate': args.hint_rate,
                       'alpha': args.alpha,
                       'iterations': args.iterations}

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
    print('gain_parameters:')
    print(gain_parameters)
    print('RMSE Performance:')
    print(rmse_dict)

    return imputed_data_x, rmse_dict


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['letter', 'spam', 'uci-secom_complete_cv_90'],
        default='uci-secom_complete_cv_90',
        type=str)
    parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        default=0.2,
        type=float)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=128,
        type=int)
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=100,
        type=float)
    parser.add_argument(
        '--iterations',
        help='number of training iterations',
        default=10000,
        type=int)

    args = parser.parse_args()

    # Calls main function
    imputed_data, rmse = main(args)