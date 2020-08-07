from csv import reader
from math import exp
from random import randrange
'''
#example of prediction using logistic regression
def predict(row,coeff):
	y = coeff[0]
	for i in range (0,len(row)-1):	
		y+=row[i]*coeff[i+1]
	return 1/(1+exp(-y));	
	
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

coeff = [-0.406605464, 0.852573316, -1.104746259]
for row in dataset:
	output = predict(row,coeff)
	print('actual=',row[-1],'predicted=',output)
	if (output<0.5):
		print('0')
	else:
		print('1')
#end of example
#example of coefficient updation 
def predict(row,coeff):
	y = coeff[0]
	for i in range (0,len(row)-1):	
		y+=row[i]*coeff[i+1]
	return 1/(1+exp(-y));	

def coefficients_sgd(train, l_rate, epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(epoch):
		#sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			#sum_error += error**2
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
		
	return coef
		
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
l_rate = 0.3
n_epoch = 100
coeff = coefficients_sgd(dataset, l_rate, n_epoch)
print(coeff)

#end of example
'''

#actual project implementation
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def str_column_to_float(dataset, column):
	for row in dataset:
		row[column]=float(row[column].strip())

def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def cross_val_split(dataset,k_fold):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / k_fold)
	for i in range(k_fold):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))

def coefficients_sgd(train, l_rate, epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(epoch):
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef


def logistic_algo(train,test,l_rate,epoch):
	predictions = list()
	coeff = coefficients_sgd(train, l_rate, epoch)
	for row in test:
		yhat = predict(row, coeff)
		yhat = round(yhat)
		predictions.append(yhat)
	return(predictions)

def accuracy_metric(predicted, actual):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def evaluate_algo(dataset,k_fold,l_rate,epoch):
	folds = cross_val_split(dataset,k_fold)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set,[])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predict = logistic_algo(train_set,test_set,l_rate,epoch)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(predict, actual)
		scores.append(accuracy)
	return scores

#loading and preparing data		
filename = 'diabetes_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)

#normalise
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

#evaluate algorithm
k_fold = 5
l_rate = 0.1
epoch = 100
score = evaluate_algo(dataset,k_fold,l_rate,epoch)
print(score)
print('mean accuracy=',sum(score)/float(k_fold))






