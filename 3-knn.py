from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#已經沒有"from sklearn.cross_validation import train_test_split"這個函式庫了，合併為上面那個函式庫
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV


train_data = pd.read_csv('1216_train_sliding.csv',index_col=0)
train_feature=train_data.drop(['1day_label','5day_label','10day_label'],axis=1)
train_target=train_data['1day_label']

test_data = pd.read_csv('1216_test_sliding.csv',index_col=0)
test_feature=test_data.drop(['1day_label','5day_label','10day_label'],axis=1)
test_target=test_data['1day_label']


model=KNeighborsClassifier(algorithm="auto",n_neighbors=9,weights="distance",p=1,leaf_size=10)


model.fit(train_feature,train_target)

pred_test=model.predict(test_feature)
print(len(test_target))
print(len(pred_test))

print("   TN FP")
print(metrics.confusion_matrix(test_target,pred_test))
print("   FN  TP")

tn, fp, fn, tp = metrics.confusion_matrix(test_target,pred_test).ravel()
print("TN=",tn,"FP=",fp,"FN=",fn,"TP=",tp)


if((tp+fp)!=0):
    precission=tp/(tp+fp)
    if((tp+fn)!=0):
        recall=tp/(tp+fn)
    else:
        recall=0
else:
    precission=0
    if((tp+fn)!=0):
        recall=tp/(tp+fn)
    else:
        recall=0

if((fp+tn)==0):
    FP_rate=0
else:
    FP_rate=fp/(fp+tn)

#F_score
if((recall+precission)!=0):
    F_score=2*((precission*recall)/(precission+recall))
else:
    F_score=0

print("accuracy   : ",round(metrics.accuracy_score(pred_test, test_target)*100 , 2) )
print("precission : ",round(precission,4)*100)
print("recall     : ",round(recall,4)*100)
print("TP-rate    : ",round(recall,4)*100)
print("FP-rate    : ",round(FP_rate,4)*100)
print("F_score    : ",round(F_score,4)*100)

pred_train=model.predict(train_feature)
print("   TN FP")
print(metrics.confusion_matrix(train_target,pred_train))
print("   FN  TP")

print("accuracy   : ",round(metrics.accuracy_score(train_target,pred_train)*100 , 2) )


train_target=train_data['5day_label']
test_target=test_data['5day_label']
model.fit(train_feature,train_target)
pred_test=model.predict(test_feature)
print("accuracy_5DAYS   : ",round(metrics.accuracy_score(train_target,pred_train)*100 , 2) )

train_target=train_data['10day_label']
test_target=test_data['10day_label']
model.fit(train_feature,train_target)
pred_test=model.predict(test_feature)
print("accuracy _10days  : ",round(metrics.accuracy_score(train_target,pred_train)*100 , 2) )