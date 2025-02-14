import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d


#parameter: 
#number: the number of generated data we needed
#method: 1:data Augmentation; 
#        2:linear / quadratic interpolation
def simulate_movement(filePath,number,method): 
    df = pd.read_csv(filePath)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')  

    if method == 1:
        df_new = data_augmentation(df, number)
    elif method == 2:
        df_new = interpolation(df, number)
    else:
        raise ValueError("Method should be 1 (Data Augmentation) or 2 (Interpolation)")
    
    return df_new

def data_augmentation(df, number):
    df_aug = df.copy()

    for _ in range(number):
        df_temp = df.sample(n=1, replace=True).copy() 
        perturbation = np.random.normal(0, 0.0001, size=(1, 2)) 
        df_temp['location_lat'] += perturbation[0, 0]
        df_temp['location_long'] += perturbation[0, 1]
        df_aug = pd.concat([df_aug, df_temp], ignore_index=True)
    
    return df_aug


def interpolation(df, number):
    
    df_interpolated = df.copy()

    num_individuals = len(df['individual_id'].unique())  
    base_points = number // num_individuals 
    extra_points = number % num_individuals  

    for idx, id in enumerate(df['individual_id'].unique()):#group by individual_id
        subset = df[df['individual_id'] == id]
        
        timestamps = subset['timestamp'].astype('int64')  # convert timestamp data to int for calculation
        latitudes = subset['location_lat'].values
        longitudes = subset['location_long'].values

        # at least 3 data for quadratic interpolation
        kind = 'quadratic' if len(subset) > 2 else 'linear'
        
        # create interpolation function
        interp_lat = interp1d(timestamps, latitudes, kind=kind, fill_value='extrapolate')
        interp_long = interp1d(timestamps, longitudes, kind=kind, fill_value='extrapolate')

        num_new_points = base_points + (1 if idx < extra_points else 0)
        
        new_timestamps = np.linspace(timestamps.min(), timestamps.max(), num_new_points)
        new_latitudes = interp_lat(new_timestamps)
        new_longitudes = interp_long(new_timestamps)
        
        # create new DataFrame
        new_subset = pd.DataFrame({
            'timestamp': pd.Series(pd.to_datetime(new_timestamps, unit='ns')).dt.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'location_lat': new_latitudes,
            'location_long': new_longitudes,
            'individual_id': id,
            'tag_id': subset['tag_id'].iloc[0],
            'dataset_id': subset['dataset_id'].iloc[0]
        })
        df_interpolated = pd.concat([df_interpolated, new_subset], ignore_index=True)
    
    return df_interpolated



if __name__ == "__main__":

    file_path = "sample_moveBank.csv"  
    number_of_points = 10
    method = 2
    
    result_df = simulate_movement(file_path, number_of_points, method)  
    result_df.to_csv("new_generated_data.csv", index=False)




#     old chunk
#     step_size=0.01 

#     original_length = len(data)
#     repeat_factor = number // original_length  
#     remainder = number % original_length  

#     new_data = pd.concat([data] * repeat_factor, ignore_index=True)  
#     new_data = pd.concat([new_data, data.sample(remainder, replace=False)], ignore_index=True)  


#     new_data['location-lat'] = new_data['location-lat'] + np.random.uniform(-step_size, step_size, len(new_data))
#     new_data['location-long'] = new_data['location-long'] + np.random.uniform(-step_size, step_size, len(new_data))
    
#     # Convert timestamp to datetime
#     new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

#     # Add random 0-60 min
#     random_minutes = np.random.randint(1, 61, size=len(new_data))  # Random int from 1 to 60
#     new_data['timestamp'] = new_data['timestamp'] + pd.to_timedelta(random_minutes, unit='m')

#     new_data['timestamp'] = new_data['timestamp'] + timedelta(hours=1)
    
#     # Convert back to string format with milliseconds
#     new_data['timestamp'] = new_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]

#     return new_data


#     old chunk 
#     # convert timestamp to datetime 
#     df['timestamp'] = pd.to_datetime(data['timestamp'])

#     # empty list，for saving the data after interpolation 
#     interpolated_data_list = []

#     # group by individual_id 
#     for individual, group in data.groupby('individual_id'):
#         # sort by timestamp
#         group = group.sort_values(by='timestamp')

#         #convert to Unix timestamp（s）
#         group['time_numeric'] = group['timestamp'].astype('int64') // 10**9

#         # at least 3 data for quadratic interpolation
#         kind = 'quadratic' if len(group) > 2 else 'linear'

#         # create interpolation function
#         latitude_interp = interp1d(group['time_numeric'], group['location_lat'], kind=kind)
#         longitude_interp = interp1d(group['time_numeric'], group['location_long'], kind=kind)

#         # generate new 20 timestamps
#         new_times = np.linspace(group['time_numeric'].min(), group['time_numeric'].max(), num=number)

#         # calculate
#         new_latitudes = latitude_interp(new_times)
#         new_longitudes = longitude_interp(new_times)

#         # generate new DataFrame
#         interpolated_group = pd.DataFrame({
#             "timestamp": pd.to_datetime(new_times, unit='ms'),
#             "location_lat": new_latitudes,
#             "location_long": new_longitudes,
#             "individual_id": individual,  # same individual_id as before
#             "tag_id": group['tag_id'].iloc[0],  # take the first tag_id of a group 
#             "dataset_id": group['dataset_id'].iloc[0]  # take the first dataset_id
#         })

        
#         interpolated_data_list.append(interpolated_group)

#     interpolated_data = pd.concat(interpolated_data_list, ignore_index=True)

#     interpolated_data = interpolated_data.sort_values(by=['individual_id', 'timestamp'])

