import pandas as pd
import numpy as np

data_x = pd.read_csv('data/Data_X.csv', index_col='ID')

def get_numeric_cols(data):
    numeric_columns = data_x.select_dtypes(include=np.number).columns
    numeric_columns.drop(['DAY_ID'])
    fr_numeric_cols = numeric_columns[numeric_columns.str[:2] != 'DE']
    de_numeric_cols = numeric_columns[numeric_columns.str[:2] != 'FR']
    return fr_numeric_cols, de_numeric_cols

fr_num_cols, de_num_cols = get_numeric_cols(data_x)


    