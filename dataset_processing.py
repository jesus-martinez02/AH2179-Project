import pandas as pd
import numpy as np
import datetime as dt
import itertools

WEATHER_PATH = "open-meteo-59.30N18.16E24m.csv" ## Path for the weather .csv file


def generate_dataset(input_path, num_past_intervals, interval_length, num_timesteps_whole_network,
                     exclude_zero_flows, start_time = dt.time(7,30),end_time = dt.time(8,30) ,sensor_subset = False,sensors=None,
                     list_sensors_up_down_stream = None):
    
    """
    Main function that performs all the data processing and adds all the features required to generate a complete
    train or test dataset.

    Parameters:
        input_path: path of the csv file where the dataset is stored
        num_past_intervals: how many intervals in the past should be considered for lagged features of the same sensor
        interval_length: length in minutes of intervals in lagged features of the same sensor
        num_timesteps_whole_network: how many intervals in the past should be considered for the lagged features of 
        ALL the sensors. This should be set to 0 if we don't want to use downstream or upstream flow features. 
        exclude_zero_flows: whether timestamps with flow = 0 are included or not in the outuput dataset
        start_time: Starting time for the output dataset (this does not affect time flow features calculaution. If for example
        7:30 is provided, it will still use data before 7:30 for the calculation of flow in the previous x minutes, but
        the first timestamp in the output dataset will be 7:30)
        end_time: Ending time for the output dataset
        sensor_subset: Bool to determine whether only a subset of the sensors wants to be incldued in the output dataset.
        For instance, this can be useful if we want to generate a dataset for only 1 or a few sensor.
        sensors: List of sensors which will be considered. Only works if sensor_subset = True.
        list_sensors_up_down_stream: List of sensors to consider for creating upstream or downstream flow features. Note that
        this is important to specify to limit the number of features in the output dataset. If nothing is provided, it
        will process these features for all the sensors, which can end up with a very large dataset

    Returns:
        Processed pandas DataFrame representing a processed training or test dataset
    
    """

    raw_df = pd.read_csv(input_path, delimiter = ";")

    if sensor_subset:
        raw_df = raw_df[raw_df["DP_ID"].isin(sensors)]

    filtered_df = raw_df[(raw_df["Interval_60"] >= start_time.hour - 1) & (raw_df["Interval_60"] <= end_time.hour + 1)]


    base_df = generate_base_dataset(filtered_df)
    full_df = pd.merge(base_df,filtered_df[["DP_ID", "Date", "Time", "SPEED_MS_AVG", "FLOW"]], how = "left", on=["DP_ID", "Date", "Time"])
    
    portal_order_df = pd.DataFrame(data=[['E4S 56,780', 5], ['E4S 56,490', 6], ['E4S 56,160', 7], ['E4S 57,055',4],
       ['E4S 58,140', 1], ['E4S 57,435',3], ['E4S 57,820',2], ['E4S 55,620',8 ]], columns = ["PORTAL", "portal_order"])

    full_df = full_df.merge(portal_order_df, how="inner", on= "PORTAL")

    full_df_transformed = transform_initial_dataset(full_df)

    sorted_df = full_df_transformed.sort_values(by=["portal_order", "DP_ID", "Date","Time" ], ascending=True).copy()
    sorted_df = sorted_df.reset_index(drop=True)

    if list_sensors_up_down_stream == None:
        list_sensors_up_down_stream = sorted_df["DP_ID"].unique()

    sorted_to_train_df = add_training_features(sorted_df, num_past_intervals, interval_length, num_timesteps_whole_network,
                                            start_time, end_time,
                                               list_sensors_up_down_stream)
    

    dummy_vars_dp_id_df = pd.get_dummies(sorted_to_train_df["DP_ID"], prefix="sensor_id_", dtype=float)
    dummy_vars_portal_df = pd.get_dummies(sorted_to_train_df["portal_order"], prefix="portal_", dtype=float)

    train_df_with_nulls = pd.concat([sorted_to_train_df, dummy_vars_dp_id_df, dummy_vars_portal_df], axis=1)
    
    if exclude_zero_flows: 
        train_df = train_df_with_nulls[train_df_with_nulls["FLOW"] > 0].copy()
        grouped_days = train_df[["Date", "Time", "DP_ID"]].groupby(["Date", "DP_ID"], as_index=False).nunique()
        valid_days_df = grouped_days[grouped_days["Time"] > 55][["Date","DP_ID"]]
        train_df = train_df.merge(valid_days_df, on=["Date","DP_ID"], how="right")
    else:
        train_df = train_df_with_nulls.copy()

    return train_df

## Function to generate a complete dataset with all the timestamps and time interval attributes (with no flow or speed attributes)
def generate_base_dataset(df):
    all_portal_time_combinations = list(itertools.product(df["DP_ID"].unique(),df["Date"].unique(),df["Time"].unique()))
    full_base_df = pd.DataFrame(all_portal_time_combinations, columns = ["DP_ID", "Date", "Time"])
    mapping_sensors_portal_df = df[["DP_ID","PORTAL"]].drop_duplicates()
    mapping_time_intervals_df = df[["Time","Interval_1", "Interval_5", "Interval_15", "Interval_30", "Interval_60"]].drop_duplicates()
    full_df_step_1 = pd.merge(full_base_df, mapping_sensors_portal_df, how = "inner", on=["DP_ID"])
    full_df_step_2 = pd.merge(full_df_step_1, mapping_time_intervals_df, how = "inner", on=["Time"])

    return full_df_step_2.copy()


## Function to perform basic transformations to the initial dataset
def transform_initial_dataset(df):
    df["FLOW"] = df["FLOW"].fillna(0)
    df["SPEED_MS_AVG"] = df["SPEED_MS_AVG"].fillna(25)
    df["speed_km_h_avg"] = df["SPEED_MS_AVG"] * 3.6
    df["flow_x_speed"] =  df["speed_km_h_avg"] * df["FLOW"]
    df["datetime_rounded_h"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str)).dt.floor("h")
    df["datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    return df.copy()



## Function to add all the features required for training to the dataset
def add_training_features(sorted_df, num_past_intervals, interval_length, num_timesteps_whole_network, start_time, end_time,
                          up_down_sensors):

    sorted_df = add_weather_data(sorted_df)

    sorted_df = add_calendar_features(sorted_df)

    # Create target feature
    window = 15
    sorted_df['flow_next_15min'] = (
        sorted_df.groupby('DP_ID')['FLOW']
        .transform(lambda x: x.rolling(window=window, min_periods=1).sum().shift(-window+1))
        )
        

    # Add lagged featues for flow and speed
    for i in range(num_past_intervals):
        sorted_df['flow_prev_' + str(i*interval_length) + "_" +  str((i+1)*interval_length) + '_min'] = (
            sorted_df.groupby('DP_ID')['FLOW']
            .transform(lambda x: x.rolling(window=interval_length, min_periods=1).sum().shift(1+i*interval_length))
            )
        
        
        sorted_df['speed_prev_' + str(i*interval_length) + "_" +  str((i+1)*interval_length) + '_min'] = (
        sorted_df.groupby('DP_ID')['flow_x_speed']
        .transform(lambda x: x.rolling(window=interval_length, min_periods=1).sum().shift(1+i*interval_length))
        ) / sorted_df['flow_prev_' + str(i*interval_length) + "_" +  str((i+1)*interval_length) + '_min']
        
    sorted_df = add_up_down_stream_flow_features(sorted_df,num_timesteps_whole_network,up_down_sensors) 
    sorted_to_train_df = sorted_df[(sorted_df['datetime'].dt.time >= start_time) & (sorted_df['datetime'].dt.time <= end_time)].copy()
    sorted_to_train_df = sorted_to_train_df.reset_index(drop=True)
    return sorted_to_train_df.copy()


## Function to add calendar features to the dataset
def add_calendar_features(sorted_df):
    dummy_vars_dow = pd.get_dummies(sorted_df["datetime"].dt.day_of_week, prefix="dow_", dtype=float)
    dummy_vars_interval = pd.get_dummies(sorted_df["Interval_15"], prefix="bin_interval_", dtype=float)
    sorted_df["is_weekend"] = sorted_df["datetime"].dt.day_of_week.isin([5,6]).astype(int)
    return pd.concat([sorted_df, dummy_vars_dow, dummy_vars_interval], axis=1)


## Function to add upstream and downstream features to the dataset
def add_up_down_stream_flow_features(sorted_df, num_timesteps_whole_network,up_down_sensors):
    flow_columns = sorted_df.filter(like="flow_prev_").columns.to_list()

    for i,flow_column in enumerate(flow_columns):
        if i == num_timesteps_whole_network:
            break
        flow_wide = sorted_df[sorted_df["DP_ID"].isin(up_down_sensors)].pivot(index="datetime", columns="DP_ID", values=flow_columns[i])
        flow_wide = flow_wide.add_prefix("flow_DP_ID_" + flow_column).reset_index()

        sorted_df = sorted_df.merge(flow_wide, on="datetime", how="left")
    

    return sorted_df.copy()

## Function to add weather data to the dataset
def add_weather_data(traffic_df):
    weather_df = pd.read_csv(WEATHER_PATH)
    weather_df["time"] = pd.to_datetime(weather_df["time"])
    merged_df = traffic_df.merge(right=weather_df, how="left", left_on=["datetime_rounded_h"], right_on=["time"])

    return merged_df.copy()
