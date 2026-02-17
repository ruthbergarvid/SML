

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb

bicycle_data = pd.read_csv("training_data_VT2026.csv", na_values='?', dtype={'ID': str}).dropna().reset_index()

##cyclic data
#hour of day
bicycle_data['hour_sin'] = np.sin(2*np.pi*bicycle_data['hour_of_day']/24)
bicycle_data['hour_cos'] = np.cos(2*np.pi*bicycle_data['hour_of_day']/24)
#day of week
bicycle_data['day_sin'] = np.sin(2*np.pi*bicycle_data['day_of_week']/7)
bicycle_data['day_cos'] = np.cos(2*np.pi*bicycle_data['day_of_week']/7)
#month
bicycle_data['month_sin'] = np.sin(2*np.pi*bicycle_data['month']/12)
bicycle_data['month_cos'] = np.cos(2*np.pi*bicycle_data['month']/12)

np.random.seed(1)


#different sample sizes
accuracy_samplesize = np.arange(500,1501,50, dtype=float)

ind = 0
for i in range(500,1501,50) :
    
    train_index = np.random.choice(bicycle_data.shape[0], i, replace= False)

    train_index_bool = bicycle_data.index.isin(train_index)
    #samma siffror fast bool variant, 1600 lång

    traindata = bicycle_data.iloc[train_index_bool]

    testdata = bicycle_data.iloc[~train_index_bool]

    testdata = testdata[0:100]
    #nu är testdata 100 lång oavsett traindata

    model_log = skl_lm.LogisticRegression(solver='lbfgs', max_iter=1000)

    X_train = traindata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth', 'windspeed', 'cloudcover', 'visibility']]

    Y_train = traindata[['increase_stock']]

    X_test = testdata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth', 'windspeed', 'cloudcover', 'visibility']]

    Y_test = testdata[['increase_stock']]


    model_log.fit(X_train, Y_train)

    predict_prob = model_log.predict_proba(X_test)

    prediction_test = np.empty(len(X_test), dtype= object)
    prediction_test = np.where(predict_prob[:,0] >=0.5, 'high_bike_demand', 'low_bike_demand')

    accuracy_samplesize[ind] = np.mean(prediction_test.squeeze() == Y_test.squeeze())
    print(np.mean(prediction_test.squeeze() == Y_test.squeeze()))
    print(accuracy_samplesize[ind])
    ind = ind+1
    

arr = np.arange(500,1501,50)
print(accuracy_samplesize)
#en array att kunna plotta mot
plt.scatter(arr, accuracy_samplesize)
plt.xlabel('Antal testdata')
plt.ylabel('precission')
plt.show()


#bra vid 1050 och 1100

#testar backwards selection

#tar bort en feature i taget
#från 15- 4



accuracy_feature = np.zeros(11)

train_index = np.random.choice(bicycle_data.shape[0], 1100, replace= False)

train_index_bool = bicycle_data.index.isin(train_index)
#samma siffror fast bool variant, 1600 lång

traindata = bicycle_data.iloc[train_index_bool]

testdata = bicycle_data.iloc[~train_index_bool]

model_log = skl_lm.LogisticRegression(solver='lbfgs', max_iter=1000)

X_train = traindata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth', 'windspeed', 'cloudcover', 'visibility']]

Y_train = traindata[['increase_stock']]

X_test = testdata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth', 'windspeed', 'cloudcover', 'visibility']]

Y_test = testdata[['increase_stock']]

model_log.fit(X_train, Y_train)

predict_prob = model_log.predict_proba(X_test)

prediction_test = np.empty(len(X_test), dtype= object)
prediction_test = np.where(predict_prob[:,0] >=0.5, 'high_bike_demand', 'low_bike_demand')

accuracy_feature[10]= np.mean(prediction_test.squeeze() == Y_test.squeeze())

####################
 
X_train = traindata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth', 'windspeed', 'cloudcover']]

X_test = testdata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth', 'windspeed', 'cloudcover']]


model_log.fit(X_train, Y_train)

predict_prob = model_log.predict_proba(X_test)

prediction_test = np.empty(len(X_test), dtype= object)
prediction_test = np.where(predict_prob[:,0] >=0.5, 'high_bike_demand', 'low_bike_demand')

accuracy_feature[9]= np.mean(prediction_test.squeeze() == Y_test.squeeze())

##############
   
X_train = traindata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth', 'windspeed']]

X_test = testdata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth', 'windspeed']]


model_log.fit(X_train, Y_train)

predict_prob = model_log.predict_proba(X_test)

prediction_test = np.empty(len(X_test), dtype= object)
prediction_test = np.where(predict_prob[:,0] >=0.5, 'high_bike_demand', 'low_bike_demand')

accuracy_feature[8]= np.mean(prediction_test.squeeze() == Y_test.squeeze())
#############
   
X_train = traindata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth']]

X_test = testdata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow', 'snowdepth']]


model_log.fit(X_train, Y_train)

predict_prob = model_log.predict_proba(X_test)

prediction_test = np.empty(len(X_test), dtype= object)
prediction_test = np.where(predict_prob[:,0] >=0.5, 'high_bike_demand', 'low_bike_demand')

accuracy_feature[7]= np.mean(prediction_test.squeeze() == Y_test.squeeze())

###############
   
X_train = traindata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow']]

X_test = testdata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity', 'precip', 'snow']]


model_log.fit(X_train, Y_train)

predict_prob = model_log.predict_proba(X_test)

prediction_test = np.empty(len(X_test), dtype= object)
prediction_test = np.where(predict_prob[:,0] >=0.5, 'high_bike_demand', 'low_bike_demand')

accuracy_feature[6]= np.mean(prediction_test.squeeze() == Y_test.squeeze())

####################

   
X_train = traindata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity', 'precip']]

X_test = testdata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity', 'precip']]


model_log.fit(X_train, Y_train)

predict_prob = model_log.predict_proba(X_test)

prediction_test = np.empty(len(X_test), dtype= object)
prediction_test = np.where(predict_prob[:,0] >=0.5, 'high_bike_demand', 'low_bike_demand')

accuracy_feature[5]= np.mean(prediction_test.squeeze() == Y_test.squeeze())

###############

   
X_train = traindata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity']]

X_test = testdata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew', 'humidity']]


model_log.fit(X_train, Y_train)

predict_prob = model_log.predict_proba(X_test)

prediction_test = np.empty(len(X_test), dtype= object)
prediction_test = np.where(predict_prob[:,0] >=0.5, 'high_bike_demand', 'low_bike_demand')

accuracy_feature[4]= np.mean(prediction_test.squeeze() == Y_test.squeeze())

#################

   
X_train = traindata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew']]

X_test = testdata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp', 'dew']]


model_log.fit(X_train, Y_train)

predict_prob = model_log.predict_proba(X_test)

prediction_test = np.empty(len(X_test), dtype= object)
prediction_test = np.where(predict_prob[:,0] >=0.5, 'high_bike_demand', 'low_bike_demand')

accuracy_feature[3]= np.mean(prediction_test.squeeze() == Y_test.squeeze())

######

X_train = traindata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp']]

X_test = testdata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp']]



model_log.fit(X_train, Y_train)

predict_prob = model_log.predict_proba(X_test)

prediction_test = np.empty(len(X_test), dtype= object)
prediction_test = np.where(predict_prob[:,0] >=0.5, 'high_bike_demand', 'low_bike_demand')

accuracy_feature[2]= np.mean(prediction_test.squeeze() == Y_test.squeeze())

#########
X_train = traindata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime']]

X_test = testdata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime']]

model_log.fit(X_train, Y_train)

predict_prob = model_log.predict_proba(X_test)

prediction_test = np.empty(len(X_test), dtype= object)
prediction_test = np.where(predict_prob[:,0] >=0.5, 'high_bike_demand', 'low_bike_demand')

accuracy_feature[1]= np.mean(prediction_test.squeeze() == Y_test.squeeze())
X_train = traindata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime']]

X_test = testdata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime']]


model_log.fit(X_train, Y_train)

predict_prob = model_log.predict_proba(X_test)

prediction_test = np.empty(len(X_test), dtype= object)
prediction_test = np.where(predict_prob[:,0] >=0.5, 'high_bike_demand', 'low_bike_demand')

accuracy_feature[0]= np.mean(prediction_test.squeeze() == Y_test.squeeze())

arr = np.arange(4,15)
print(accuracy_feature)
#en array att kunna plotta mot
plt.plot(arr, accuracy_feature)
plt.xlabel('Number of features')
plt.ylabel('Precission')
plt.show()

#bäst med 7
#--> mina features är ['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp']

#testar threshhold

train_index = np.random.choice(bicycle_data.shape[0], 1100, replace= False)

train_index_bool = bicycle_data.index.isin(train_index)
#samma siffror fast bool variant, 1600 lång

traindata = bicycle_data.iloc[train_index_bool]

testdata = bicycle_data.iloc[~train_index_bool]

model_log = skl_lm.LogisticRegression(solver='lbfgs', max_iter=1000)

X_train = traindata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp']]

Y_train = traindata[['increase_stock']]

X_test = testdata[['hour_sin', 'hour_cos', 'day_sin','day_cos', 'month_sin', 'month_cos', 'holiday', 'weekday', 'summertime', 'temp']]

Y_test = testdata[['increase_stock']]

model_log.fit(X_train, Y_train)

predict_prob = model_log.predict_proba(X_test)

accuracy_tresh = np.zeros(100)

for i in range(0,100):

    tresh = 0.01*i
    prediction_test = np.empty(len(X_test), dtype= object)
    prediction_test = np.where(predict_prob[:,0] >=tresh, 'high_bike_demand', 'low_bike_demand')
    accuracy_tresh[i] = np.mean(prediction_test.squeeze() == Y_test.squeeze())

plt.plot(np.linspace(0,1,100), accuracy_tresh)
plt.xlabel('treshold')
plt.ylabel('precission')
plt.show()

print(max(accuracy_tresh))

##best accuracy på tresh = 0.4

#1100 samples, trashold 0,4, 7 features

