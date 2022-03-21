# Simple Linear Regression on the Diabetes Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
 
# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Split a dataset into a train and test set
def train_test_split(dataset, split):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy
 
# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)
 
# Evaluate an algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, split, *args):
	train, test = train_test_split(dataset, split)
	test_set = list()
	for row in test:
		row_copy = list(row)
		row_copy[-1] = None
		test_set.append(row_copy)
	predicted,coef= algorithm(train, test_set, *args)
	actual = [row[-1] for row in test]
	rmse = rmse_metric(actual, predicted)
	return rmse,coef
 
# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))
 
# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar
 
# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])
 
# Calculate coefficients
def coefficients(dataset):
	x1 = [row[0] for row in dataset]
	x2 = [row[1] for row in dataset]
	x3 = [row[2] for row in dataset]
	x4 = [row[3] for row in dataset]
	x5 = [row[4] for row in dataset]

	y = [row[5] for row in dataset]
	m1, m2, m3, m4, m5, my = mean(x1), mean(x2), mean(x3), mean(x4), mean(x5), mean(y),
	b1 = covariance(x1, m1, y, my) / variance(x1, m1)
	b2 = covariance(x2, m2, y, my) / variance(x2, m2)
	b3 = covariance(x3, m3, y, my) / variance(x3, m3)
	b4 = covariance(x4, m4, y, my) / variance(x4, m4)
	b5 = covariance(x5, m5, y, my) / variance(x5, m5)
	b0 = my - b1 * m1 - b2 * m2 - b3 * m3 - b4 * m4 - b5 * m5
	return [b0, b1, b2, b3, b4, b5]
 
 
# Simple linear regression algorithm
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 , b2, b3, b4, b5 = coefficients(train)
	coefficient = [b0, b1 , b2, b3, b4, b5]
	for row in test:
		yhat = b0 + b1 * row[0] + b2 * row[1] + b3 * row[2] + b4 * row[3] + b5 * row[4]
		predictions.append(yhat)
	return predictions,coefficient

def predict(bp,i,bmi,dpf,age,coef):
	output = coef[0] + coef[1] * bp + coef[2] * i + coef[3] * bmi +coef[4]*dpf + coef[5]*age
	if output >0.5:
		return 1
	else:
		return 0



# multiple linear regression on diabetes dataset
seed(1)
# load and prepare data
filename = 'diabetes.csv'
dataset = load_csv(filename)
print(dataset[0])
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
split = 0.6
rmse, coef= evaluate_algorithm(dataset, simple_linear_regression, split)
print('RMSE: %.3f' % (rmse))

print("output: ",predict(94,0,49.3,0.358,27,coef))

#BloodPressure,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome