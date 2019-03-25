#Deep Neural Network

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

#Fitting Classifier DNN to the Training Set
from sklearn.neural_network import MLPClassifier
start_training = time.time()
classifier = MLPClassifier(hidden_layer_sizes=(27,27,27),max_iter=500)
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
cm_rf = confusion_matrix(y_test.ravel(), y_pred.ravel())

#Calculate accuracy_score
from sklearn.metrics import accuracy_score
accuracy_rf = accuracy_score(y_test, y_pred)
accuracy_rf_normalize = accuracy_score(y_test, y_pred, normalize=False)

#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_kfold =cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10, n_jobs=-1)
accuracies_kfold_mean = accuracies_kfold.mean()
accuracies_kfold_std = accuracies_kfold.std()

'''
#Test Set Normal
from matplotlib.colors import ListedColormap
X1 = X_test[:,0] #frame_len
Y1 = y_test[:,0] #normal
Y2 = y_pred[:,0] #predicted_normal-traffic

plt.plot(X1, Y1)
plt.title('K_Nearest Neighbor (Training set)')
plt.xlabel('Traffic')
plt.ylabel('Status')
plt.legend()
plt.show