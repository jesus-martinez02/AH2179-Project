import numpy as np
import pandas as pd
import datetime as dt


from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras
from sklearn.cluster import KMeans


## Function to train Linear Regression on all sensors, returning the trained model and performance
## metrics on the train and the test datasets

def train_linear_regression(train_df,test_df):
    interval_features = train_df.filter(like="bin_interval_").columns.to_list()
    flow_features = train_df.filter(regex=r"^flow_prev").columns.to_list()
    dummy_vars_dow = train_df.filter(like="dow_").columns.to_list()
    dummy_vars_dp_id = train_df.filter(like="sensor_id").columns.to_list()
    dummy_vars_portal = train_df.filter(like="portal__").columns.to_list()

    train_features = flow_features + dummy_vars_dow + ["is_weekend"] + interval_features + \
                        dummy_vars_dp_id + dummy_vars_portal

    target = "flow_next_15min"


    X_train = train_df[train_features].to_numpy()
    y_train = train_df[target].to_numpy()
    X_test = test_df[train_features].to_numpy()
    y_test = test_df[target].to_numpy()
        
    model = LinearRegression()
    model.fit(X_train,y_train)

    y_train_pred = model.predict(X_train)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

    return [model, rmse_train, rmse_test, mape_train,mape_test]


## Function to train LASSO Regression on all sensors, returning the trained model and performance
## metrics on the train and the test datasets
def train_lasso_regression(train_df,test_df, alpha=0.1):
    interval_features = train_df.filter(like="bin_interval_").columns.to_list()
    flow_features = train_df.filter(regex=r"^flow_prev").columns.to_list()
    dummy_vars_dow = train_df.filter(like="dow_").columns.to_list()
    dummy_vars_dp_id = train_df.filter(like="sensor_id").columns.to_list()
    dummy_vars_portal = train_df.filter(like="portal__").columns.to_list()

    train_features = flow_features + dummy_vars_dow + ["is_weekend"] + interval_features + \
                        dummy_vars_dp_id + dummy_vars_portal

    target = "flow_next_15min"


    X_train = train_df[train_features].to_numpy()
    y_train = train_df[target].to_numpy()
    X_test = test_df[train_features].to_numpy()
    y_test = test_df[target].to_numpy()
        
    model = Lasso(alpha=alpha)
    model.fit(X_train,y_train)

    y_train_pred = model.predict(X_train)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

    return [model, rmse_train, rmse_test, mape_train,mape_test]



## Function to train Linear Regression on individual sensors, returning the trained models and performance
## metrics on the train and the test datasets
def train_sensor_models_with_lasso(train_df,test_df, alpha=0.1):
    y_pred_train = []
    y_pred_test = []

    DP_ID_values = train_df["DP_ID"].unique()
    model_dict = dict.fromkeys(DP_ID_values)

    for dp_id in model_dict.keys():
        red_train_df = train_df[train_df["DP_ID"] == dp_id].copy()
        red_test_df = test_df[test_df["DP_ID"] == dp_id].copy()

        interval_features = red_train_df.filter(like="bin_interval_").columns.to_list()
        flow_features = red_train_df.filter(regex=r"^flow_prev").columns.to_list()
        dummy_vars_dow = red_train_df.filter(like="dow_").columns.to_list()

        train_features = train_features = flow_features + dummy_vars_dow + ["is_weekend"] + interval_features
        target = "flow_next_15min"

        X_train = red_train_df[train_features].to_numpy()
        y_train = red_train_df[target].to_numpy()
        X_test = red_test_df[train_features].to_numpy()
        y_test = red_test_df[target].to_numpy()
                
        model = Lasso(alpha=alpha)
        model.fit(X_train,y_train)
        model_dict[dp_id] = model

        y_pred_train.append(model.predict(X_train).ravel())
        y_pred_test.append(model.predict(X_test).ravel())
    
    y_pred_train = np.concatenate(y_pred_train)
    y_pred_test = np.concatenate(y_pred_test)

    sorted_train_df = train_df.sort_values(by=["DP_ID","datetime"],ascending=True)
    rmse_train = (root_mean_squared_error(sorted_train_df[target].to_numpy(), y_pred_train))
    mape_train = (mean_absolute_percentage_error(sorted_train_df[target].to_numpy(), y_pred_train))


    sorted_test_df = test_df.sort_values(by=["DP_ID","datetime"],ascending=True)
    rmse_test = (root_mean_squared_error(sorted_test_df[target].to_numpy(), y_pred_test))
    mape_test = (mean_absolute_percentage_error(sorted_test_df[target].to_numpy(), y_pred_test))


    return [model_dict, rmse_train, rmse_test, mape_train, mape_test]


## Function to train Neural on all sensors, returning the trained model and performance
## metrics on the train and the test datasets
def train_nn_model(train_df,test_df, model_name = None):
    interval_features = train_df.filter(like="bin_interval_").columns.to_list()
    flow_features = train_df.filter(regex=r"^flow_prev").columns.to_list()
    dummy_vars_dow = train_df.filter(like="dow_").columns.to_list()
    dummy_vars_dp_id = train_df.filter(like="sensor_id").columns.to_list()
    dummy_vars_portal = train_df.filter(like="portal__").columns.to_list()



    train_features = flow_features + dummy_vars_dow + ["is_weekend"] + interval_features + \
                        dummy_vars_dp_id + dummy_vars_portal

    target = "flow_next_15min"

    X_train = train_df[train_features].to_numpy()
    y_train = train_df[target].to_numpy()

    X_test = test_df[train_features].to_numpy()
    y_test = test_df[target].to_numpy()


    model = Sequential()


    model.add(Dense(32, activation='relu',input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='mse', 
                metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    #model.summary()

    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    y_train_pred = model.predict(X_train, verbose=0)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)

    y_test_pred = model.predict(X_test,verbose=0)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

    if model_name is not None:
        model.save(model_name)
    return [model, rmse_train, rmse_test, mape_train,mape_test]


## Function train LASSO Regression on all sensors using a clustering approach, returning the trained models and performance
## metrics on the train and the test datasets
def train_clustering_with_lasso(train_df,test_df):

    flow_features = train_df.filter(regex=r"^flow_prev").columns.to_list()

    train_df_15_min_interval = train_df[train_df["datetime"].dt.minute.isin([0,15,30,45])][["DP_ID","datetime","flow_next_15min"]]

    pivot_df = train_df_15_min_interval.pivot_table(index="DP_ID", columns="datetime",values="flow_next_15min")
    pivot_df = pivot_df.reset_index()

    X = pivot_df.drop(columns="DP_ID").fillna(0).values

    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X)

    pivot_df["cluster"] = labels
    cluster_df = pivot_df[["DP_ID","cluster"]]

    y_pred_train = []
    y_pred_test = []

    model_dict = dict.fromkeys(np.sort(cluster_df["cluster"].unique()))

    for cluster in np.sort(cluster_df["cluster"].unique()):

        list_dp_id = np.sort(cluster_df[cluster_df["cluster"]==cluster]["DP_ID"].to_list())
        red_train_df = train_df[train_df["DP_ID"].isin(list_dp_id)].copy()
        red_test_df = test_df[test_df["DP_ID"].isin(list_dp_id)].copy()

        interval_features = red_train_df.filter(like="bin_interval_").columns.to_list()
        flow_features = red_train_df.filter(regex=r"^flow_prev").columns.to_list()
        dummy_vars_dow = red_train_df.filter(like="dow_").columns.to_list()


        train_features = train_features = flow_features + dummy_vars_dow + ["is_weekend"] + interval_features


        target = "flow_next_15min"



        X_train = red_train_df[train_features].to_numpy()
        y_train = red_train_df[target].to_numpy()

        X_test = red_test_df[train_features].to_numpy()
        y_test = red_test_df[target].to_numpy()

        
        model = Lasso(alpha=0.1)
        model.fit(X_train,y_train)

        model_dict[cluster] = model

        for dp_id in list_dp_id:
            X_train = red_train_df[red_train_df["DP_ID"] == dp_id][train_features].to_numpy()
            y_train = red_train_df[red_train_df["DP_ID"] == dp_id][target].to_numpy()
            X_test = red_test_df[red_test_df["DP_ID"] == dp_id][train_features].to_numpy()
            y_test = red_test_df[red_test_df["DP_ID"] == dp_id][target].to_numpy()


            y_pred_train.append(model.predict(X_train).ravel())
            y_pred_test.append(model.predict(X_test).ravel())

    

    y_pred_train = np.concatenate(y_pred_train)
    y_pred_test = np.concatenate(y_pred_test)    

    train_df_clusters = train_df.merge(cluster_df, on="DP_ID", how = "inner")
    test_df_clusters = test_df.merge(cluster_df, on="DP_ID", how = "inner")

    sorted_train_df = train_df_clusters.sort_values(by=["cluster","DP_ID", "datetime"], ascending=True)

    sorted_test_df = test_df_clusters.sort_values(by=["cluster","DP_ID", "datetime"], ascending=True)


    rmse_train = (root_mean_squared_error(sorted_train_df[target].to_numpy(), y_pred_train))
    mape_train = (mean_absolute_percentage_error(sorted_train_df[target].to_numpy(), y_pred_train))

    rmse_test = (root_mean_squared_error(sorted_test_df[target].to_numpy(), y_pred_test))
    mape_test = (mean_absolute_percentage_error(sorted_test_df[target].to_numpy(), y_pred_test))


    return [model_dict, rmse_train, rmse_test, mape_train, mape_test]