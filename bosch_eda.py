import os
import sys

import numpy as np
import pandas as pd

os.getcwd()
base_path = os.getcwd()
data_path = os.path.join(base_path, '[10] data')
bosch_data = os.path.join(data_path, 'bosch')
bosch_train_data = pd.read_csv(os.path.join(bosch_data, 'train_numeric.csv'))

print('Shape of data: %s' % str(bosch_train_data.shape))
print('Column names of data: %s' %(bosch_train_data.columns))
print('Type of columns: %s' %(bosch_train_data.dtypes))

bosch_train_data.isna().sum(axis = 1) / bosch_train_data.shape[0]

np.unique(bosch_train_data.isna().sum(axis = 1) / bosch_train_data.shape[0], return_counts = True)

