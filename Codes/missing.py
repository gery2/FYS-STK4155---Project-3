#Project 3 - penguins dataset
# Load libraries
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# get data
dataset = read_csv('penguins.csv')

# replacing NaN with zeros
dataset = dataset.fillna(0)

# removing the rows with only zeros
dataset = dataset.drop(dataset.index[339])
dataset = dataset.drop(dataset.index[3])

# assigning a number for island
lb = LabelEncoder()
dataset["island"] = lb.fit_transform(dataset["island"])

# transforming dataframe to array and changing the . to zero
array = dataset.values
array[335][6] = 0

# a list of penguins with missing genders
missing = []



# copy penguins with missing gender to another array
for i in range(342):
    if array[i][6] == 0:
        missing.append(array[i][:])
# remove penguins with missing gender from original array
array = np.delete(array, [7, 8, 9, 10, 46, 245, 285, 323, 335], axis=0)
missing = np.array(missing)

X = array[:,1:6]
y = array[:,6]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
'''
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='auto')))
models.append(('FFNN', MLPClassifier(solver='sgd', max_iter=10000)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
'''

# predicting the missing genders

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
#print('CART accuracy score: ', accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))

missing_X = missing[:,1:6]
predicted_X = model.predict(missing_X)


# allocate the predicted genders to the penguins with missing genders
for i in range(9):
    missing[i][6] = predicted_X[i]

# add the now labeled penguins to the rest of the penguin group
array = array.tolist(); missing = missing.tolist()
for i in range(9):
    array.append(missing[i][:])

from pandas import DataFrame

df = DataFrame(array)
# MALE = 1, FEMALE = 0
df[6] = lb.fit_transform(df[6])

array = df.values
X = array[:,1:7]
y = array[:,0]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
'''
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='auto')))
models.append(('FFNN', MLPClassifier(solver='sgd', max_iter=10000)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
'''

# Make predictions on validation dataset
model = LogisticRegression(solver='liblinear', multi_class='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
print('LR accuracy score: ', accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
print('CART accuracy score: ', accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

