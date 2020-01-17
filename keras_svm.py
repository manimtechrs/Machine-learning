# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:09:16 2019

@author: user
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data=pd.read_csv('train.csv')
data.head()
test_data=pd.read_csv('test.csv')
test_data.head()
x=data.drop(['D_StructureSystem_Precast_Concrete','D_StructureSystem_Brick_wall_with_Wooden_pillar',
           'D_StructureSystem_Wood_or_Synthetic_Resins',
           'D_sms','D_sm1','D_tm0','D_Ra_Capacity','D_Ra_CDR',
           'D_StructureSystem_SRC','D_floorTAGround'],axis=1)
test=test_data.drop(['D_StructureSystem_Precast_Concrete','D_StructureSystem_Brick_wall_with_Wooden_pillar',
           'D_StructureSystem_Wood_or_Synthetic_Resins',
           'D_sms','D_sm1','D_tm0','D_Ra_Capacity','D_Ra_CDR',
           'D_StructureSystem_SRC','D_floorTAGround'],axis=1)

X=x.drop('D_isR',axis=1)
Y=x.D_isR
#missing values in data has to be fill before training by imputer method 

from sklearn.preprocessing import Imputer
Xvalue=X.values
print(Xvalue.shape)
testvalues=test.values

#test_datavalue=test_data.values

imputer = Imputer()
transformed_X= imputer.fit_transform(Xvalue)
transformed_test= imputer.fit_transform(testvalues)

#transformed_test_data=imputer.fit_transform(test_datavalue)

print(np.isnan(transformed_X).sum())
#print(np.isnan(transformed_test_data).sum())

#scale dataset 

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(transformed_X)
test_scale = min_max_scaler.fit_transform(transformed_test)
#scale_test_data= min_max_scaler.fit_transform(transformed_test_data)

#split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scale, Y, test_size=.3)
X_scale.shape

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dropout
from keras import regularizers

model= Sequential([
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(48,)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
])

'''
input_dim=X_train.shape[1]


model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(1))
model.add(Activation('softmax'))
'''






'''

model=Sequential([
        Dense(32,activation='relu',input_shape=(48,)),
        Dense(32,activation='relu'),
        Dense(1,activation='sigmoid'),
        ])
'''
model.compile(optimizer ='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train,y_train,
               batch_size=32,epochs=100,
               validation_data=(X_test,y_test))

print(model.evaluate(X_test,y_test)[1])
'''
#test data

test_data=pd.read_csv('test.csv')
test_data.head()


test.head()
from sklearn.preprocessing import Imputer
imputer=Imputer()
print(np.isnan(transformed_test).sum())

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

'''
pred=model.predict(transformed_test)
print(pred)

import numpy as np
import pandas as pd
#np.savetxt("6/11.txt",pred,delimiter=",")



pd.DataFrame(np.asarray(pred)).to_csv('66.csv')
