#Logistic Regression 

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

#Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
start_training = time.time()
classifier_logistic_normal = LogisticRegression()
classifier_logistic_normal.fit(X_train, y_train[:,0])
classifier_logistic_icmp = LogisticRegression()
classifier_logistic_icmp.fit(X_train, y_train[:,1])
classifier_logistic_tcp = LogisticRegression()
classifier_logistic_tcp.fit(X_train, y_train[:,2])
classifier_logistic_udp = LogisticRegression()
classifier_logistic_udp.fit(X_train, y_train[:,3])
end_training = time.time()

#Predicting 
start_testing = time.time()
###Predicting the Test set result for NORMAL###
y_pred_normal = classifier_logistic_normal.predict(X_test)
###Predicting the Test set result for ICMP Echo Flood###
y_pred_icmp = classifier_logistic_icmp.predict(X_test)
###Predicting the Test set result for TCP Xmas Flood###
y_pred_tcp = classifier_logistic_tcp.predict(X_test)
###Predicting the Test set result for UDP Flood###
y_pred_udp = classifier_logistic_udp.predict(X_test)
y_pred_logistic = np.column_stack((y_pred_normal,y_pred_icmp,y_pred_tcp,y_pred_udp))
end_testing = time.time()
training_time = end_training - start_training
testing_time = end_testing - start_testing

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm_logistic = confusion_matrix(y_test.ravel(), y_pred_logistic.ravel())

#Calculate accuracy_score
from sklearn.metrics import accuracy_score
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
accuracy_logistic_normalize = accuracy_score(y_test, y_pred_logistic, normalize=False)

#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_kfold_normal =cross_val_score(estimator=classifier_logistic_normal, X = X_train, y = y_train[:,0], cv = 10, n_jobs=-1)
accuracies_kfold_icmp = cross_val_score(estimator=classifier_logistic_icmp, X = X_train, y = y_train[:,1], cv = 10, n_jobs=-1)
accuracies_kfold_tcp = cross_val_score(estimator=classifier_logistic_tcp, X = X_train, y = y_train[:,2], cv = 10, n_jobs=-1)
accuracies_kfold_udp = cross_val_score(estimator=classifier_logistic_udp, X = X_train, y = y_train[:,3], cv = 10, n_jobs=-1)
accuracies_kfold = np.concatenate((accuracies_kfold_normal,accuracies_kfold_icmp,accuracies_kfold_tcp,accuracies_kfold_udp), axis=0)
accuracies_kfold_mean = accuracies_kfold.mean()
accuracies_kfold_std = accuracies_kfold.std()


'''
#Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01 ),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01 ))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0,75, cmap = ListedColormap(('red','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1],
                c = ListedColormap(('red','blue'))(i), label = j)
plt.title('K_Nearest Neighbor (Training set)')
plt.xlabel('Traffic')
plt.ylabel('Status')
plt.legend()
plt.show

#Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01 ),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01 ))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0,75, cmap = ListedColormap(('red','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1],
                c = ListedColormap(('red','blue'))(i), label = j)
plt.title('K_Nearest Neighbor (Test set)')
plt.xlabel('Traffic')
plt.ylabel('Status')
plt.legend()
plt.show
'''

'''
arp_count_normal = 0
for i in range (0, len(dataset)):
    if dataset_array[i,4]==0 and dataset_array[i,25]==0:
        arp_count_normal +=1

