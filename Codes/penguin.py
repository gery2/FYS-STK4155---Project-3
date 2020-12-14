#Project 3 - penguins dataset
# Load libraries
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

d1 = dataset[['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm']]
#sns.boxplot(data=d1, width=0.5,fliersize=5)
#pyplot.show()

'''
# scatterplot
sns.pairplot(dataset, hue="species", height=2,diag_kind="hist")
pyplot.show()



# culmen_depth vs culmen_length
sns.FacetGrid(dataset, hue="species", height=7) \
   .map(pyplot.scatter, "culmen_length_mm", "culmen_depth_mm") \
   .add_legend()
pyplot.show()



# culmen_depth vs flipperlength
sns.FacetGrid(dataset, hue="species", height=7) \
   .map(pyplot.scatter, "culmen_length_mm", "flipper_length_mm") \
   .add_legend()
pyplot.show()



# Flipperlength distribution (best visual)
sns.violinplot(x="species", y="flipper_length_mm", data=dataset,size=7)
pyplot.show()



# KDEPlot of flipper length
sns.FacetGrid(dataset, hue="species", height=5,) \
   .map(sns.kdeplot, "flipper_length_mm",shade=True) \
   .add_legend()
pyplot.show()



# Flipper length vs. body mass
sns.FacetGrid(dataset, hue="species", height=7) \
   .map(pyplot.scatter, "body_mass_g", "flipper_length_mm") \
   .add_legend()
pyplot.show()
'''



# Split-out validation dataset
array = dataset.values
X = array[:,2:6]
y = array[:,0]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

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





























































































#
