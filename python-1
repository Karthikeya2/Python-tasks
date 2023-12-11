# Python-tasks 1.py

import pandas as pd

import numpy as np

data = pd.read_csv("dataset-1.csv")

df = pd.DataFrame(data)


def generate_car_matrix(df):

    matrix = pd.pivot_table(df, values='car', index='id_1', columns='id_2', fill_value=0).values

    np.fill_diagonal(matrix, 0)
    
    result_matrix = pd.DataFrame(matrix, index=df['id_1'].unique(), columns=df['id_2'].unique())

    return result_matrix
df = pd.read_csv('dataset-1.csv')
result_matrix = generate_car_matrix(df)

print(result_matrix)
