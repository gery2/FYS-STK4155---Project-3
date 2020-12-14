#Project 3 - penguins dataset - CART
# Load libraries
import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# get data
dataset = read_csv('penguins.csv')

# shape
#print(dataset.shape)
# head
#print(dataset.head(20))

# Handling missing values
from sklearn.impute import SimpleImputer
#setting strategy to 'most frequent' to impute by the mean
imputer = SimpleImputer(strategy='most_frequent')# strategy can also be mean or median
dataset.iloc[:,:] = imputer.fit_transform(dataset)
#print(dataset.isnull().sum())

# MALE = 2, FEMALE = 1
lb = LabelEncoder()
dataset["sex"] = lb.fit_transform(dataset["sex"])
#print(dataset['sex'][:5])
#print(dataset['species'].value_counts())


# Split-out validation dataset
array = dataset.values
X = array[:,2:6]
y = array[:,0]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Make predictions on validation dataset
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
print('CART accuracy score: ', accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


CART = []

for i in range(1000):
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=i)

    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    # Evaluate predictions
    CART.append(accuracy_score(Y_validation, predictions))


print('CART -mean- accuracy score: ', np.mean(CART))














#
