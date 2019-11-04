import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from IPython import embed
def read_data(path):
    data=pd.read_csv(path)
    return data

def convert_df_to_dataset(df,labels,features):
    for feature in features:
        if df[feature].dtype is not 'int64' or df[feature].dtype is not 'float64':
            df[feature]=df[feature].astype('category').cat.codes
    df_list=features+labels
    df.columns.values.tolist()
    df=df[df_list]
    dataset = df.values
    X = dataset[:,0:len(features)]
    Y = dataset[:,len(features):len(features)+len(labels)]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scale = min_max_scaler.fit_transform(X)
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    return X_train,X_val,X_test,Y_train,Y_val,Y_test

def prepare_prediction_dataset(df,features):
        for feature in features:
            if df[feature].dtype is not 'int64' or df[feature].dtype is not 'float64':
                df[feature]=df[feature].astype('category').cat.codes
        df_list=features
        df.columns.values.tolist()
        df=df[df_list]
        dataset = df.values
        X = dataset[:,0:len(features)]
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scale = min_max_scaler.fit_transform(X)
        return X_scale
