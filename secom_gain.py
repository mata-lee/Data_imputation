#%%
#%%
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

from data_loader import data_loader
from gain import gain
from utils import rmse_loss

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
      - imputed_data_x: imputed data
      - rmse: Root Mean Squared Error
    '''

    data_name = args.data_name
    miss_rate = args.miss_rate

    gain_parameters = {'batch_size': args.batch_size,
                       'hint_rate': args.hint_rate,
                       'alpha': args.alpha,
                       'iterations': args.iterations}

    # Load data and introduce missingness
    ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)

    # Impute missing data
    imputed_data_x = gain(miss_data_x, gain_parameters)

    # Report the RMSE performance
    rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)

    print()
    print('RMSE Performance: ' + str(np.round(rmse, 4)))

    return imputed_data_x, rmse
#%%
if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['letter', 'spam'],
        default='spam',
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
#%%

import pandas as pd

pd.DataFrame(miss_data_x).to_csv(os.path.join(os.getcwd(), '[10] data/' + data_name + '_miss' + '.csv'), index = False)
pd.DataFrame(imputed_data_x).to_csv(os.path.join(os.getcwd(), '[10] data/' + data_name + '_imp_gain' + '.csv'), index = False)
pd.DataFrame(data_m).to_csv(os.path.join(os.getcwd(), '[10] data/' + data_name + '_miss_mask' + '.csv'), index = False)

# np.loadtxt(os.path.join(os.getcwd(), '[10] data/' + data_name + '.csv'), delimiter=",", skiprows=1)
# np.loadtxt(os.path.join(os.path.join(os.getcwd(), '[10] data/' + data_name + '_miss' + '.csv'), delimiter=",", skiprows=1)
# np.loadtxt(os.path.join(os.path.join(os.getcwd(), '[10] data/' + data_name + '_miss' + '.csv'), delimiter=",", skiprows=1)
# np.loadtxt(os.path.join(os.path.join(os.getcwd(), '[10] data/' + data_name + '_miss' + '.csv'), delimiter=",", skiprows=1)


rmse_gain = rmse


#%%
# Mean imputation
from sklearn.impute import SimpleImputer

med_imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
med_imputer = med_imputer.fit(miss_data_x)
imputed_data_med = med_imputer.transform(miss_data_x)

# Report the RMSE performance
rmse_med = rmse_loss(ori_data_x, imputed_data_med, data_m)

print()
print('RMSE Performance: ' + str(np.round(rmse_med, 4)))


#%%
# EM imputation
import impyute as impy

data_missing = pd.DataFrame(miss_data_x)
em_imputed = impy.em(miss_data_x)

rmse_em = rmse_loss(ori_data_x, em_imputed, data_m)

print()
print('RMSE Performance: ' + str(np.round(rmse_em, 4)))

pd.DataFrame(imputed_data_med).to_csv(os.path.join(os.getcwd(), '[10] data/' + data_name + '_imp_med' + '.csv'), index = False)
pd.DataFrame(em_imputed).to_csv(os.path.join(os.getcwd(), '[10] data/' + data_name + '_imp_EM' + '.csv'), index = False)

# RMSE
# GAIN: 0.0905
# median imputation: 0.1095
# EM imputation: 0.1453
