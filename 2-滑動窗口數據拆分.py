from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

data = pd.read_csv('1216_sliding.csv')

data_train=data.drop(['Date','1day_label','5day_label','10day_label'],axis=1)
x_list=['1day_label','5day_label','10day_label']


pre_train_data,pre_test_data,train_label,test_label=train_test_split(data_train,data[x_list],test_size=0.2,random_state=4)


train_data=pd.DataFrame(pre_train_data)
test_data=pd.DataFrame(pre_test_data)

new=pd.concat([train_data,train_label],axis='columns')
new.to_csv("1216_train_sliding.csv")

new=pd.concat([test_data,test_label],axis='columns')
new.to_csv("1216_test_sliding.csv")

