#Decision Tree

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time 

#importing dataset
dataset = pd.read_csv('rawdata_selected.csv')
###Taking care of missing value/ Missing value = 0
dataset = dataset.fillna(0)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 25].values

##Encoding port
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_port = LabelEncoder()
X[:,4] = labelencoder_port.fit_transform(X[:,4])
onehotencoder = OneHotEncoder(categorical_features=[4])
X = onehotencoder.fit_transform(X).toarray()

#Encoding y (0= normal, 1=icmp flood attack, 2=tcp xmas flood, 3=udp flood )
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
y = np.reshape(y, (-1,1))
onehotencoder_y = OneHotEncoder(categorical_features=[0])
y = onehotencoder_y.fit_transform(y).toarray()

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting Classifier Decision Tree to the Training Set
from sklearn.tree import DecisionTreeClassifier
start_training = time.time()
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
end_training = time.time()

#predicting the test set result
start_testing = time.time()
y_pred = classifier.predict(X_test)
end_testing = time.time()

training_time = end_training - start_training
testing_time = end_testing - start_testing

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm_dt = confusion_matrix(y_test.ravel(), y_pred.ravel())

#Calculate accuracy_score
from sklearn.metrics import accuracy_score
accuracy_dt = accuracy_score(y_test, y_pred)
accuracy_dt_normalize = accuracy_score(y_test, y_pred, normalize=False)

#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_kfold =cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10, n_jobs=-1)
accuracies_kfold_mean = accuracies_kfold.mean()
accuracies_kfold_std = accuracies_kfold.std()

#####################TESTBED###################################################
testbed = pd.read_csv('testbed_selected.csv')
testbed = testbed.fillna(0)

X_testbed = testbed.iloc[:, :-1].values
y_testbed = testbed.iloc[:, 25].values

import random 
counter = 5
new_X = X_testbed[1,:]
new_y = y_testbed[1]

for x in range(counter):
    rand_normal = random.randint(0,1000)
    rand_icmp = random.randint(1000,2000)
    rand_tcp = random.randint(2000,3000)
    rand_udp = random.randint(3000,4001)
    new_X = np.vstack((X_testbed[rand_normal,:].T, X_testbed[rand_icmp,:].T, X_testbed[rand_tcp,:].T, X_testbed[rand_udp,:].T, new_X))
    new_y = np.vstack((y_testbed[rand_normal], y_testbed[rand_icmp], y_testbed[rand_tcp], y_testbed[rand_udp], new_y))


#encoding X
new_X = onehotencoder.fit_transform(new_X).toarray()    
#encoding Y
new_y = onehotencoder_y.fit_transform(new_y).toarray()


y_pred_testbed = classifier.predict(new_X)
accuracy_dt_testbed = accuracy_score(new_y,y_pred_testbed)
accuracy_dt_normalize = accuracy_score(new_y, y_pred_testbed, normalize=False)
accuracy_kfold_testbed = cross_val_score(estimator=classifier,X=new_X, y=new_Y, cv=10, n_jobs=1)
accuracy_kfold_testbed_mean = accuracy_kfold_testbed.mean()
