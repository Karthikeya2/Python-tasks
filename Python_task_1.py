"""Python Task 1
"""Question 1: Car Matrix Generation

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


"""Question 2: Car Type Count Calculation

Python 1-2nd task
import pandas as pd
import numpy as np
data = pd.read_csv("F.csv")
df = pd.DataFrame(data)

def get_type_count(df):
    df['car_type'] = pd.cut(df['car'], bins=[float('-inf'), 15, 25, float('inf')],
                    labels=['low', 'medium', 'high'], right=False)

    type_count = df['car_type'].value_counts().to_dict()
    type_count = dict(sorted(type_count.items()))
    
    return type_count
    
df = pd.read_csv('F.csv')

result_type_count = get_type_count(df)
print(result_type_count)

"""Question 3: Bus Count Index Retrieval

import pandas as pd
import numpy as np
data = pd.read_csv("dataset-1.csv")

df = pd.DataFrame(data)
def get_bus_indexes(df):
    bus_mean = df['bus'].mean()
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
    bus_indexes.sort()
    return bus_indexes
    
df = pd.read_csv('dataset-1.csv')
result_bus_indexes = get_bus_indexes(df)
print(result_bus_indexes)


"""Question 4: Route Filtering

import pandas as pd
import numpy as np
data = pd.read_csv("dataset-1.csv")

df = pd.DataFrame(data)

def filter_routes(df):
    route_avg_truck = df.groupby('route')['truck'].mean() 
    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()
    filtered_routes.sort()
    return filtered_routes
    
df = pd.read_csv('dataset-1.csv')
result_filtered_routes = filter_routes(df)

print(result_filtered_routes)

"""Question 5: Matrix Value Modification

import pandas as pd
import numpy as np

def generate_car_matrix(df):
    matrix = pd.pivot_table(df, values='car', index='id_1', columns='id_2', fill_value=0).values
    np.fill_diagonal(matrix, 0)
    return pd.DataFrame(matrix, index=df['id_1'].unique(), columns=df['id_2'].unique())
    
def modify_matrix(matrix_df):
    modified_matrix = matrix_df.copy()
    modified_matrix = np.where(modified_matrix > 20, modified_matrix * 0.75, modified_matrix * 1.25)
    modified_matrix = np.round(modified_matrix, 1)
    return pd.DataFrame(modified_matrix, index=matrix_df.index, columns=matrix_df.columns)
    
df = pd.read_csv('dataset-1.csv')
car_matrix = generate_car_matrix(df)
modified_matrix = modify_matrix(car_matrix)
print(modified_matrix)


