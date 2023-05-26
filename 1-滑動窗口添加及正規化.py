from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing


data=pd.read_csv("1216.csv")

#data=data_orign.drop(['1day_label','5day_label','10day_label'],axis=1)

def add_label_1(dataframe, column):
    label = []
    label.append(0)
    for i in range(len(dataframe)):
        if(i>0):
            label.append(dataframe[column][i-1])
    return pd.Series(label)
data["Open-1"] = add_label_1(data, "Open")
data["High-1"] = add_label_1(data, "High")
data["Low-1"] = add_label_1(data, "Low")
data["Close-1"] = add_label_1(data, "Close")
data["Adj Close-1"] = add_label_1(data, "Adj Close")
data["Volume-1"] = add_label_1(data, "Volume")

def add_label_2(dataframe, column):
    label = []
    for i in range(2):
        label.append(0)
    for i in range(len(dataframe)):
        if(i>1):
            label.append(dataframe[column][i-2])
    return pd.Series(label)
data["Open-2"] = add_label_2(data, "Open")
data["High-2"] = add_label_2(data, "High")
data["Low-2"] = add_label_2(data, "Low")
data["Close-2"] = add_label_2(data, "Close")
data["Adj Close-2"] = add_label_2(data, "Adj Close")
data["Volume-2"] = add_label_2(data, "Volume")

def add_label_3(dataframe, column):
    label = []
    for i in range(3):
        label.append(0)
    for i in range(len(dataframe)):
        if(i>2):
            label.append(dataframe[column][i-3])
    return pd.Series(label)
data["Open-3"] = add_label_3(data, "Open")
data["High-3"] = add_label_3(data, "High")
data["Low-3"] = add_label_3(data, "Low")
data["Close-3"] = add_label_3(data, "Close")
data["Adj Close-3"] = add_label_3(data, "Adj Close")
data["Volume-3"] = add_label_3(data, "Volume")

def add_label_4(dataframe, column):
    label = []
    for i in range(4):
        label.append(0)
    for i in range(len(dataframe)):
        if(i>3):
            label.append(dataframe[column][i-4])
    return pd.Series(label)
data["Open-4"] = add_label_4(data, "Open")
data["High-4"] = add_label_4(data, "High")
data["Low-4"] = add_label_4(data, "Low")
data["Close-4"] = add_label_4(data, "Close")
data["Adj Close-4"] = add_label_4(data, "Adj Close")
data["Volume-4"] = add_label_4(data, "Volume")



pre_normalize=data.drop(['Date'],axis=1)

normalize__scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
normalize__data_array=normalize__scaler.fit_transform(pre_normalize)
normalize__data=pd.DataFrame(normalize__data_array)

new_data=pd.concat([data['Date'],normalize__data],axis='columns')
new_data=new_data.drop(new_data.index[[0,1,2,3,4,5,6]])

new_data=new_data.reset_index(drop=True)

label_data=pd.read_csv('label_1216.csv')

x_list=['1day_label','5day_label','10day_label']
output=pd.concat([new_data,label_data[x_list]],axis='columns')

#print(output)

output.to_csv('1216_sliding.csv')
