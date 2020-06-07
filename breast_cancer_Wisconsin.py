# import libraries
import numpy as np
import pandas as pd
#import matplotlib.plotly as plt

#get dataset
dataset = pd.read_csv('breast_cancer.csv')
print(dataset.head(5))
X = dataset.iloc[:, :-1]
y= dataset.iloc[:, -1]

#Splitting Data into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state = 0)

#transform Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#Train logistic regression on Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#Predict result on test set
y_pred = classifier.predict(X_test)

#Confusion Metrics
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
ac = accuracy_score(y_test, y_pred)
print(ac)

#Computing accuracy with Kfold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10)
accuracy_avg = accuracies.mean()*100
accuracy_deviation = accuracies.std()*100
print("Avg Accuracy : {:.2f}%".format(accuracy_avg))
print("Standard Deviation : {:.2f}%".format(accuracy_deviation))

