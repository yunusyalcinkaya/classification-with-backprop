# back propagation algorithm

from random import randrange
from random import random
from csv import reader
from math import exp
import numpy as np
import tensorflow as tf
from PIL import Image
import os


def preprocess(path, names, csv_file, image_class):
    with open(names, 'a') as file:
        for image in os.listdir(path):
            if image.endswith('.jpg') or image.endswith('.png') or image.endswith('.jpeg'):
                file.write(image + '\n')
    with open(names, 'r') as file:
        for line in file:
            line = line.rstrip("\n")
            image = Image.open(path + line)
            image = image.resize((10, 10))
            image.save(path + line)
            image_content = tf.io.read_file(path + line)
            image_content = tf.io.decode_image(image_content, channels=1)
            image_content = np.array(image_content)
            image_content = image_content.ravel()
            image_content = np.append(image_content, image_class)
            with open(csv_file, 'a') as file:
                np.savetxt(file, image_content.reshape(-1, image_content.shape[-1]), delimiter=',')


preprocess('/content/chicken/', 'chicken_names', 'image_dataset.csv', 0)
preprocess('/content/cow/', 'cow_names', 'image_dataset.csv', 1)
preprocess('/content/spider/', 'spider_names', 'image_dataset.csv', 2)


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
def str_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset):
    minmax = [[min(column), max(column)] for column in zip(*dataset)]
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def trainset_split(dataset, n_split):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_split)
    for i in range(n_split):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def evaluate_algorithm(dataset, back_propagation, n_split, *args):
    splitted = trainset_split(dataset, n_split)
    train_set = list(splitted)
    train_set.remove(splitted[2])
    train_set = sum(train_set, [])
    test_set = list()

    for row in splitted[2]:
        row_copy = list(row)
        test_set.append(row_copy)

    predicted = back_propagation(train_set, test_set, *args)
    actual = [row[-1] for row in splitted[2]]
    print("beklenen: ", actual)
    print("tahmin: ", predicted)
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    accuracy = correct / float(len(actual)) * 100.0
    return accuracy


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    sigmoid = 1.0 / (1.0 - exp(-activation))
    return sigmoid


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            neuron['output'] = activate(neuron['weights'], inputs)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * neuron['output'] * (1.0 - neuron['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print("epoch: ", epoch, "  neurons: ", network[-1])


def initialize_network(n_inputs, n_hidden_neuron, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden_neuron)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden_neuron + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def back_propagation(train, test, l_rate, n_epoch, n_hidden_neuron):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden_neuron, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return (predictions)


# load and prepare data
filename = 'image_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    str_to_float(dataset, i)
# convert class column to integers
str_to_int(dataset, len(dataset[0]) - 1)
# normalize input variables
normalize_dataset(dataset)
# evaluate algorithm
n_split = 5
l_rate = 0.5
n_epoch = 100  # normali 500
n_hidden_neuron = 2
accuracy = evaluate_algorithm(dataset, back_propagation, n_split, l_rate, n_epoch, n_hidden_neuron)
print('Scores: %s' % accuracy)#back propagation algorithm

from random import randrange
from random import random
from csv import reader
from math import exp
import numpy as np
import tensorflow as tf
from PIL import Image
import os

def preprocess(path,names,csv_file,image_class):
  with open(names,'a') as file:
    for image in os.listdir(path):
      if image.endswith('.jpg') or image.endswith('.png') or image.endswith('.jpeg'):
        file.write(image + '\n')
  with open(names,'r') as file:
    for line in file:
      line = line.rstrip("\n")
      image = Image.open(path + line)
      image = image.resize((10,10))
      image.save(path + line)
      image_content = tf.io.read_file(path + line)
      image_content = tf.io.decode_image(image_content,channels=1)
      image_content = np.array(image_content)
      image_content = image_content.ravel()
      image_content = np.append(image_content,image_class)
      with open(csv_file,'a') as file:
        np.savetxt(file,image_content.reshape(-1,image_content.shape[-1]), delimiter=',')

preprocess('/content/chicken/', 'chicken_names', 'image_dataset.csv',0)
preprocess('/content/cow/', 'cow_names', 'image_dataset.csv',1)
preprocess('/content/spider/', 'spider_names', 'image_dataset.csv',2)

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
def str_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())



# Convert string column to integer
def str_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset):
	minmax = [[min(column), max(column)] for column in zip(*dataset)]
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def trainset_split(dataset, n_split):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_split)
	for i in range(n_split):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def evaluate_algorithm(dataset, back_propagation, n_split, *args):
	splitted = trainset_split(dataset, n_split)
	train_set = list(splitted)
	train_set.remove(splitted[2])
	train_set = sum(train_set, [])
	test_set = list()

	for row in splitted[2]:
		row_copy = list(row)
		test_set.append(row_copy)

	predicted = back_propagation(train_set, test_set, *args)
	actual = [row[-1] for row in splitted[2]]
	print("beklenen: ", actual)
	print("tahmin: ",predicted)
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	accuracy = correct / float(len(actual)) * 100.0
	return accuracy

def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	sigmoid = 1.0 / (1.0 - exp(-activation))
	return sigmoid

def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			neuron['output'] = activate(neuron['weights'], inputs)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(neuron['output'] - expected[j])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * neuron['output'] * (1.0 - neuron['output'])

def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] -= l_rate * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print("epoch: ", epoch, "  neurons: ",network[-1])

def initialize_network(n_inputs, n_hidden_neuron, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden_neuron)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden_neuron + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

def back_propagation(train, test, l_rate, n_epoch, n_hidden_neuron):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden_neuron, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)


# load and prepare data
filename = 'image_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_to_float(dataset, i)
# convert class column to integers
str_to_int(dataset, len(dataset[0])-1)
# normalize input variables
normalize_dataset(dataset)
# evaluate algorithm
n_split = 5
l_rate = 0.5
n_epoch = 100 #normali 500
n_hidden_neuron = 2
accuracy = evaluate_algorithm(dataset, back_propagation, n_split, l_rate, n_epoch, n_hidden_neuron)
print('Scores: %s' % accuracy)