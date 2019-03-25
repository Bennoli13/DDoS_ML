#Kernel Support Vector Machine (Kernel SVM)

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

#Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
start_training = time.time()
classifier_ksvm_normal = SVC(kernel = 'rbf', random_state=0)
classifier_ksvm_normal.fit(X_train, y_train[:,0])
classifier_ksvm_icmp = SVC(kernel = 'rbf', random_state=0)
classifier_ksvm_icmp.fit(X_train, y_train[:,1])                              
classifier_ksvm_tcp = SVC(kernel = 'rbf', random_state=0)
classifier_ksvm_tcp.fit(X_train, y_train[:,2])
classifier_ksvm_udp = SVC(kernel = 'rbf', random_state=0)
classifier_ksvm_udp.fit(X_train, y_train[:,3])
end_training = time.time()

#Predicting
start_testing = time.time()
###Predicting the Test set result for NORMAL###
y_pred_normal = classifier_ksvm_normal.predict(X_test)
###Predicting the Test set result for ICMP Echo Flood###
y_pred_icmp = classifier_ksvm_icmp.predict(X_test)
###Predicting the Test set result for TCP Xmas Flood###
y_pred_tcp = classifier_ksvm_tcp.predict(X_test)
###Predicting the Test set result for UDP Flood###
y_pred_udp = classifier_ksvm_udp.predict(X_test)
y_pred_ksvm = np.column_stack((y_pred_normal,y_pred_icmp,y_pred_tcp,y_pred_udp))
end_testing= time.time()

training_time = end_training - start_training
testing_time = end_testing - start_testing

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm_ksvm = confusion_matrix(y_test.ravel(), y_pred_ksvm.ravel())

#Calculate accuracy_score
from sklearn.metrics import accuracy_score
accuracy_logistic = accuracy_score(y_test, y_pred_ksvm)
accuracy_logistic_normalize = accuracy_score(y_test, y_pred_ksvm, normalize=False)


#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_kfold_normal =cross_val_score(estimator=classifier_ksvm_normal, X = X_train, y = y_train[:,0], cv = 10, n_jobs=-1)
accuracies_kfold_icmp = cross_val_score(estimator=classifier_ksvm_icmp, X = X_train, y = y_train[:,1], cv = 10, n_jobs=-1)
accuracies_kfold_tcp = cross_val_score(estimator=classifier_ksvm_tcp, X = X_train, y = y_train[:,2], cv = 10, n_jobs=-1)
accuracies_kfold_udp = cross_val_score(estimator=classifier_ksvm_udp, X = X_train, y = y_train[:,3], cv = 10, n_jobs=-1)
accuracies_kfold = np.concatenate((accuracies_kfold_normal,accuracies_kfold_icmp,accuracies_kfold_tcp,accuracies_kfold_udp), axis=0)
accuracies_kfold_mean = accuracies_kfold.mean()
accuracies_kfold_std = accuracies_kfold.std()