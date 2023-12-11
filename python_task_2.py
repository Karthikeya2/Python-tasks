Task 2 -Question 1
import pandas as pd

def calculate_distance_matrix(dataset_3_path):

    dataset_3_df = pd.read_csv(dataset_3_path)

    unique_ids = dataset_3_df.columns

    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids)
    distance_matrix = distance_matrix.fillna(0)

    
    for _, row in dataset_3_df.iterrows():
        for i in range(len(unique_ids)):
            for j in range(i + 1, len(unique_ids)):
                start_id, end_id = unique_ids[i], unique_ids[j]

               
                if pd.notna(row[start_id]) and pd.notna(row[end_id]):
                    # Update distance_matrix with bidirectional distances
                    distance_matrix.at[start_id, end_id] += row[end_id]
                    distance_matrix.at[end_id, start_id] += row[end_id]

    return distance_matrix

distance_matrix = calculate_distance_matrix('dataset-3.csv')
print(distance_matrix)


Task 2-Question -2 

import pandas as pd
def unroll_distance_matrix(distance_matrix):  
    unrolled_distances = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    for start_id in distance_matrix.index:
        for end_id in distance_matrix.columns:
           if start_id == end_id:
                continue
            distance = distance_matrix.at[start_id, end_id]
            unrolled_distances = unrolled_distances.append({
                'id_start': start_id,
                'id_end': end_id,
                'distance': distance
            }, ignore_index=True)

    return unrolled_distances
distance_matrix = calculate_distance_matrix('dataset-3.csv')
unrolled_distances = unroll_distance_matrix(distance_matrix)
print(unrolled_distances)

Task2-Question-3

import pandas as pd

def find_ids_within_ten_percentage_threshold(unrolled_distances, reference_value):
    
    reference_rows = unrolled_distances[unrolled_distances['id_start'] == reference_value]

    print("Reference Rows:")
    print(reference_rows)

    avg_distance = reference_rows['distance'].mean()

    lower_bound = avg_distance - (avg_distance * 0.1)
    upper_bound = avg_distance + (avg_distance * 0.1)

    within_threshold = reference_rows[(reference_rows['distance'] >= lower_bound) & (reference_rows['distance'] <= upper_bound)]

    result_ids = within_threshold['id_start'].unique() 
    result_ids.sort()
    return result_ids

reference_value = 801 
result_ids = find_ids_within_ten_percentage_threshold(unrolled_distances, reference_value)


print(result_ids)

Task 2-Question 4:

import pandas as pd

def calculate_toll_rate(distances_df):
    
    distances_df['moto'] = distances_df['distance'] * 0.8
    distances_df['car'] = distances_df['distance'] * 1.2
    distances_df['rv'] = distances_df['distance'] * 1.5
    distances_df['bus'] = distances_df['distance'] * 2.2
    distances_df['truck'] = distances_df['distance'] * 3.6

    return distances_df

result_with_toll_rates = calculate_toll_rate(unrolled_distances)
print(result_with_toll_rates)

Task 2-Question-5

import pandas as pd
import numpy as np
import os
print(os.getcwd())
df_dataset_2 = pd.read_csv(dataset-2)
import datetime


def calculate_time_based_toll_rates(df):
   
    df['timestamp'] = pd.to_datetime(df['timestamp'])

   
    df['day'] = df['timestamp'].dt.day_name()
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute

  
    def calculate_discount_factor(row):
        if row['day'] in ['Saturday', 'Sunday']:
            return 0.7  
        elif 0 <= row['hour'] < 10 or 18 <= row['hour'] <= 23:
            return 0.8  
        elif 10 <= row['hour'] < 18:
            return 1.2 
        else:
            return 1.0  

   
    df['discount_factor'] = df.apply(calculate_discount_factor, axis=1)

  
    vehicle_types = ['moto', 'car', 'rv', 'bus', 'truck']
    for vehicle_type in vehicle_types:
        df[vehicle_type + '_rate'] = df[vehicle_type] * df['discount_factor']

    
    df = df.drop(['day', 'hour', 'minute', 'discount_factor'], axis=1)

    return df


result_with_time_based_toll_rates = calculate_time_based_toll_rates(df_dataset_2)
print(result_with_time_based_toll_rates.head())


