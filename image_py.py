from __future__ import print_function
from PIL import Image
from glob import glob
import pandas as pd
from IPython import embed
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np





def convert_images_to_training_dataset(features,labels):
    X_scale = np.array(features).astype(np.float32)
    Y = np.array(labels)
    for i in range(len(X_scale)): # map pixel data between 0 and 1
        X_scale[i]=list(map(lambda x: round(float(x/255),3), X_scale[i]))
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    return X_train,X_val,X_test,Y_train,Y_val,Y_test

def convert_images_to_prediction_dataset(features):
        if len(features)>0:
            X_scale = [[x[0],np.array(x[1]).astype(np.float32)] for x in features]
            for i in range(len(X_scale)): # map pixel data between 0 and 1
                X_scale[i][1]=list(map(lambda x: round(float(x/255),3), X_scale[i][1])) # X_scale[i][1] : pixel value , X_scale[i][0] image_name
            return X_scale
        else:
            no_prediction_file_exception()
