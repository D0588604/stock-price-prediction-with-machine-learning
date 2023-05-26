from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

data = pd.read_csv('adj_label_2049.csv')

x_list = ["Open", "High", "Low", "Close","Adj Close", "Volume"]
y_list = ["1day_label","5day_label","10day_label"]

normalize__scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
normalize__data_array=normalize__scaler.fit_transform(data[x_list])
normalize__data=pd.DataFrame(normalize__data_array)



pre_train_data,pre_test_data,train_label,test_label=train_test_split(normalize__data,data[y_list],test_size=0.2,random_state=4)

train_data=pd.DataFrame(pre_train_data)
test_data=pd.DataFrame(pre_test_data)



new=pd.concat([train_data,train_label],axis='columns')
new.to_csv("train_adj_normalize_2049.csv")

new=pd.concat([test_data,test_label],axis='columns')
new.to_csv("test_adj_normalize_2049.csv")
