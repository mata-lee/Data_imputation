#%%
# DataLoader
# Necessary packages
import os
import sys
code_path = 'C:/Users/kt NexR/Desktop/mata/work/OJT/[11] code'
if code_path not in sys.path:
    sys.path.append('C:/Users/kt NexR/Desktop/mata/work/OJT/[11] code')
import numpy as np
from utils import binary_sampler
from keras.datasets import mnist

data_name = 'uci-secom_complete_cv_90'
miss_rate = 0.2


def data_loader(data_name, miss_rate):
    '''Loads datasets and introduce missingness.

    Args:
      - data_name: letter, spam, or mnist
      - miss_rate: the probability of missing components

    Returns:
      data_x: original data
      miss_data_x: data with missing values
      data_m: indicator matrix for missing components
    '''

    # Load data
    if data_name in ['letter', 'spam', 'uci-secom_complete_cv_90']:
        file_name = os.path.join(os.getcwd(), '[10] data/' + data_name + '.csv')
        data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
    elif data_name == 'mnist':
        (data_x, _), _ = mnist.load_data()
        data_x = np.reshape(np.asarray(data_x), [60000, 28 * 28]).astype(float)

    # Parameters
    no, dim = data_x.shape

    # Introduce missing data
    data_m = binary_sampler(1 - miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m
#%%
data_name = 'uci-secom_complete_cv_90'
miss_rate = 0.2

data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)

#%%
# GAIN
'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
# import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index

# data_x = miss_data_x
gain_parameters = {'miss_rate': 0.2, 'batch_size': 128, 'hint_rate': 0.9, 'alpha': 100, 'iterations': 10000}

def gain(data_x, gain_parameters):
    '''Impute missing values in data_x

    Args:
      - data_x: original data with missing values
      - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations

    Returns:
      - imputed_data: imputed data
    '''
    # Define mask matrix
    data_m = 1 - np.isnan(data_x)

    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    # Other parameters
    no, dim = data_x.shape

    # Hidden state dimensions
    h_dim = int(dim)

    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    ## GAIN architecture
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape=[None, dim])
    # Mask vector
    M = tf.placeholder(tf.float32, shape=[None, dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape=[None, dim])

    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))  # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    ## GAIN functions
    # Generator
    def generator(x, m):
        # Concatenate Mask and Data
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        return G_prob

    # Discriminator
    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    ## GAIN structure
    # Generator
    G_sample = generator(X, M)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                  + (1 - M) * tf.log(1. - D_prob + 1e-8))

    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))

    MSE_loss = \
        tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    ## GAIN solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Start Iterations
    for it in tqdm(range(iterations)):
        # Sample batch
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]
        # Sample random vectors
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        # Sample hint vectors
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                                  feed_dict={M: M_mb, X: X_mb, H: H_mb})
        _, G_loss_curr, MSE_loss_curr = \
            sess.run([G_solver, G_loss_temp, MSE_loss],
                     feed_dict={X: X_mb, M: M_mb, H: H_mb})

    ## Return imputed data
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = data_m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]

    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    # Rounding
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data
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

# from data_loader import data_loader
from gain import gain
from utils import rmse_loss

# ori_data_x = data_x
# imputed_data_x = imputed_data

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