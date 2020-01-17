# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:19:10 2019

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


build=pd.read_csv("train.csv")

print("build dimension:{}".format(build.shape))
print(build.shape)
#df_build=pd.DataFrame(np.c_[build['data'],build['target']],columns=np.append(build['Index'],['target']))
#df_build.head()


build.head(2)
build.shape
build.info()
build.get_dtype_counts()
#build.isnull().any()

'''
build=build.drop(['D_StructureSystem_RC','D_StructureSystem_Precast_Concrete','D_StructureSystem_Steel',
            'D_StructureSystem_Reinfored_Brick','D_StructureSystem_Brick_Wall','D_StructureSystem_Brick_wall_with_Wooden_pillar',
            'D_StructureSystem_Wood_or_Synthetic_Resins','D_StructureSystem_SRC','D_StructureSystem_NaN','D_structure_6',
            'D_structure_5','D_structure_4','D_structure_3','D_structure_2','D_structure_1','D_floor','D_floorTA','D_floorTAGround',
            'D_1floorCorridorCol','D_1floorCorridorColA','D_1floorClassCol','D_1floorClassColA','D_1floorInsideCol',
            'D_1floorInsideColA','D_X4brickwall','D_X3brickwall','D_YRCwallA','D_Y4brickwall','D_Y3brickwall','D_basintype',
            'D_475Acc','D_I','D_Demand','D_Tx','D_Ty','D_sds','D_sd1','D_td0','D_sms','D_sm1','D_tm0','D_windows',
            'D_patitionwall','D_nonstructure','D_CLlarge','D_CLsmall','D_MaxCl','D_NeutralDepth','D_Ra_Capacity',
            'Total_LiveLoad','Total_Floor','Avg_Confc','Avg_MBfy','Avg_stify','Total_DeadLoad','Total_Height'],axis =1)
'''
build=build.drop(['D_StructureSystem_Precast_Concrete','D_StructureSystem_Brick_wall_with_Wooden_pillar',
           'D_StructureSystem_Wood_or_Synthetic_Resins',
           'D_sms','D_sm1','D_tm0','D_Ra_Capacity','D_Ra_CDR',
           'D_StructureSystem_SRC','D_floorTAGround'],axis =1,inplace=True)

print(build.shape)
build.info()

#clean data 
build['D_floorTAGround']=build.D_floorTAGround.fillna(value=0)
build['D_basintype']=build.D_basintype.fillna(value=0)
build['D_475Acc']=build.D_475Acc.fillna(value=0)
build['D_I']=build.D_I.fillna(value=0)
build['D_Demand']=build.D_Demand.fillna(value=0)
build['D_Tx']=build.D_Tx.fillna(value=0)
build['D_Ty']=build.D_Ty.fillna(value=0)
build['D_sds']=build.D_sds.fillna(value=0)
build['D_sd1']=build.D_sd1.fillna(value=0)
build['D_td0']=build.D_td0.fillna(value=0)
build['D_MaxCl']=build.D_MaxCl.fillna(value=0)
build['D_NeutralDepth']=build.D_NeutralDepth.fillna(value=0)
build['Total_Height']=build.Total_Height.fillna(value=0)
build['Total_DeadLoad']=build.Total_DeadLoad.fillna(value=0)
build['Total_LiveLoad']=build.Total_LiveLoad.fillna(value=0)
build['Total_Floor']=build.Total_Floor.fillna(value=0)
build['Avg_Confc']=build.Avg_Confc.fillna(value=0)
build['Avg_MBfy']=build.Avg_MBfy.fillna(value=0)
build['Avg_stify']=build.Avg_stify.fillna(value=0)
build['D_Ra_CDR']=build.D_Ra_CDR.fillna(value=0)

build.isnull().values.any()
build.info()
#build[build.isnull().any(axis=1).shape]
build.head(2)
build.shape

'''
#build[build.isnull().any(axis=1).shape]
build.head(2)
build.shape
build=build.dropna()
build=build.reset_index(drop=True)
build.shape
build.head()

'''
#Data split for training and testing
from sklearn.model_selection import train_test_split

y=build.D_isR
X=build.drop('D_isR',axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

X_train.shape, y_train.shape

X_test.shape, y_test.shape

X.columns

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
print(classifier)

Y_pred = classifier.predict(X_test)
print(Y_pred)
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)
print(cm)
print(classification_report(y_test, Y_pred))


testx=pd.read_csv('test.csv')
testx['D_floorTAGround']=testx.D_floorTAGround.fillna(value=0)
testx['D_basintype']=testx.D_basintype.fillna(value=0)
testx['D_475Acc']=testx.D_475Acc.fillna(value=0)
testx['D_I']=testx.D_I.fillna(value=0)
testx['D_Demand']=testx.D_Demand.fillna(value=0)
testx['D_Tx']=testx.D_Tx.fillna(value=0)
testx['D_Ty']=testx.D_Ty.fillna(value=0)
testx['D_sds']=testx.D_sds.fillna(value=0)
testx['D_sd1']=testx.D_sd1.fillna(value=0)
testx['D_td0']=testx.D_td0.fillna(value=0)
testx['D_MaxCl']=testx.D_MaxCl.fillna(value=0)
testx['D_NeutralDepth']=testx.D_NeutralDepth.fillna(value=0)
testx['Total_Height']=testx.Total_Height.fillna(value=0)
testx['Total_DeadLoad']=testx.Total_DeadLoad.fillna(value=0)
testx['Total_LiveLoad']=testx.Total_LiveLoad.fillna(value=0)
testx['Total_Floor']=testx.Total_Floor.fillna(value=0)
testx['Avg_Confc']=testx.Avg_Confc.fillna(value=0)
testx['Avg_MBfy']=testx.Avg_MBfy.fillna(value=0)
testx['Avg_stify']=testx.Avg_stify.fillna(value=0)
testx['D_Ra_CDR']=testx.D_Ra_CDR.fillna(value=0)
testx=testx.drop(['D_sms','D_sm1','D_tm0','D_Ra_Capacity'],axis =1)

testpredict=classifier.predict(testx)
print(testpredict)
import numpy as np
import pandas as pd
prediction = pd.DataFrame(testpredict, columns=['label']).to_csv('prediction_svm.csv')
