#KNN_ML

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state = 1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
#Fitting Classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
start_train = time.time()
classifier.fit(X_train, y_train)
end_train = time.time()
training_time = end_train - start_train

#Predicting the Test set results
start_test = time.time()
y_pred = classifier.predict(X_test)
end_test = time.time()
testing_time = end_test - start_test

#Calculate accuracy_score
from sklearn.metrics import accuracy_score
accuracy_knn = accuracy_score(y_test, y_pred)
accuracy_knn_normalize = accuracy_score(y_test, y_pred, normalize=False)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.ravel(), y_pred.ravel())

#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_kfold =cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10, n_jobs=-1)
accuracies_kfold_mean = accuracies_kfold.mean()
accuracies_kfold_std = accuracies_kfold.std()



