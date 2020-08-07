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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle

# Load dataset
file = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(file, names=names)

# summarising data
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('class').size())

#visualising data
#univariate
dataset.plot(kind='line', subplots=True, layout=(2,2), sharex=False, sharey=False)
dataset.hist(histtype='step')

#multivariate
scatter_matrix(dataset)
pyplot.show()

#splitting dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1)

#check with various models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
names =[]
results = []

#k-fold cross validation is used to predict best model
for name, model in models:
  kfold = StratifiedKFold(10,True,1)
  cv_results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  print(name,cv_results)
  print('%s:%f %f'%(name,cv_results.mean(),cv_results.std()))

pyplot.boxplot(results,names)
pyplot.title('each model result')
pyplot.xlabel('names')
pyplot.ylabel('result')
pyplot.show()

#make prediction
model = SVC(gamma='auto')
model.fit(X_train,Y_train)

#save model to file
filename = 'random_forest.pb'
pickle.dump(model,open(filename,'wb'))

#evaluate predictions

loaded_model = pickle.load(open(filename,'rb'))
predictions = loaded_model.score(X_test,Y_test)
print(accuracy_score(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
