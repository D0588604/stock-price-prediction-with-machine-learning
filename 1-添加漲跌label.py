from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

data = pd.read_csv('上銀2049.csv')



def add_1day_label(dataframe, column):
    label = []
    for i in range(len(dataframe)):
        if (i==1219):
            label.append(0)
            continue
        if (dataframe[column][i+1] >= dataframe[column][i]):
            label.append(1)
        else :
            label.append(0)
    return pd.Series(label)

def add_5day_label(dataframe, column):
    label = []
    for i in range(len(dataframe)):
        if (i==1219):
            label.append(1)
            continue
        elif(i==1218):
            label.append(1)
            continue
        elif (i==1217):
            label.append(1)
            continue
        elif (i==1216):
            label.append(1)
            continue
        elif (i==1215):
            label.append(1)
            continue
        if (dataframe[column][i+5] >= dataframe[column][i]):
            label.append(1)
        else :
            label.append(0)
    return pd.Series(label)

def add_10day_label(dataframe, column):
    label = []
    for i in range(len(dataframe)):
        if (i==1219):
            label.append(1)
            continue
        elif (i==1218):
            label.append(0)
            continue
        elif (i==1217):
            label.append(1)
            continue
        elif (i==1216):
            label.append(1)
            continue
        elif (i==1215):
            label.append(1)
            continue
        elif (i==1214):
            label.append(1)
            continue
        elif (i==1213):
            label.append(1)
            continue
        elif (i==1212):
            label.append(1)
            continue
        elif (i==1211):
            label.append(1)
            continue
        elif (i==1210):
            label.append(1)
            continue
        if (dataframe[column][i+10] >= dataframe[column][i]):
            label.append(1)
        else :
            label.append(0)
    return pd.Series(label)

data["1day_label"] = add_1day_label(data, "Close")
data["5day_label"] = add_5day_label(data, "Close")
data["10day_label"] = add_10day_label(data, "Close")

data.to_csv("adj_label_2049.csv")

