import numpy as np 
import os
import glob
from scipy.io import wavfile
from scipy import signal
from scipy.stats import zscore
import features
from features import*
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import csv
import matplotlib.pyplot as plt

##=============== DATA EXTRACTION PART ==================

def read_data(zero_padding=True, normalize=True):
	main_dir = os.getcwd()
	filename = "*/*.wav"
	files = []
	labels = []
	words = []
	length = []
	SR = 0

	for file in glob.glob(filename):
		files.append(file)
		labels.append(file.split("/")[0])
		SR, audio = wavfile.read(file)
		words.append(audio)
		length.append(len(audio))

	len_max_array = max(length)
	len_min_array = min(length)

	if(zero_padding):
		for i in range(len(words)):
			diff = len_max_array - len(words[i])
			if(diff>0):
				words[i] = np.pad( words[i] ,(int(diff/2),int(diff-diff/2)))
	if(normalize):
		words = [zscore(word) for word in words]

	for i,label in enumerate(labels):
		if label == "THE":
			labels[i]= 0.0
		if label == "A":
			labels[i]= 1.0
		if label == "TO":
			labels[i]= 2.0
		if label == "OF":
			labels[i]= 3.0	
		if label == "IN":
			labels[i]= 4.0	
		if label == "ARE":
			labels[i]= 5.0	
		if label == "AND":
			labels[i]= 6.0	
		if label == "IS":
			labels[i]= 7.0	
		if label == "THAT":
			labels[i]= 8.0	
		if label == "THEY":
			labels[i]= 9.0

	labels = np.array(labels)

	return words, labels, SR, len_max_array

class_names = ["THE","A","TO","OF","IN","ARE","AND","IS","THAT","THEY"]
win_size = 10
words,labels,SR,len_max_array = read_data()
# print(len(words))



###================== FEATURE EXTRACTION PART ==================================

def compute_stft():
	""" Computes the Short Time FOurier Transform of the audio data extracted from read_data()
		Returns: Sample frequencies, Time segments and magnitude STFT of the input data as numpy array
		NOTE:	
		Input data is Descrete Time data. So, don't be confused with the continunous time data if you plot the magnitude/power of the data 
	"""
	stft_data = []
	freq_samples = []
	seg_times = []
	window = signal.kaiser(256, beta=25)

	for i in range(0, len(words)):
		f,t,Zxx = signal.stft(words[i], SR, window, nperseg=len(window))
		freq_samples.append(f)
		seg_times.append(t)
		stft_data.append(np.abs(Zxx))

	return np.array(freq_samples), np.array(seg_times), np.array(stft_data)

f, t, Zxx = compute_stft()
# plt.pcolormesh(t[index], f[index], np.abs(Zxx[index]), vmin=0, vmax=250)
# plt.title('STFT')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

def makeCSV(train_data, test_data, train_label, test_label):
	""" Makes a .csv data file containing data and labels. 
	(Don't need to close the file with with open(filename). HAHAHA Yettikae birsinca ki vanera lekhdeko) """
	with open('stft_data.csv','w') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter = ',')
		csv_writer.writerow(train_data)
		csv_writer.writerow(test_data)
		csv_writer.writerow(train_label)
		csv_writer.writerow(test_label)

		
def get_audioData():
	""" Splits the features extracted from compute_temporal_features(), compute_spectral_features() or compute_stft() into train and test
		In the same way, splits labels into train and test. Both with 20% test size
		Returns: Test data, Train data, Train label and Test label
	"""

	# f = Features(words, labels, win_size, SR,len_max_array)
	# feat = (f.compute_spectral_features())
	freq_samples, seg_times, stft_data = compute_stft()
	X_train, X_test, Y_train, Y_test = train_test_split( stft_data, labels, test_size = 0.2)
	return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = get_audioData()
# makeCSV(X_train, X_test, Y_train, Y_test)

# print(X_train.shape)
# print(X_test.shape)



###================ UTILITIES FOR MODEL TRAINING =============================

class Flatten(nn.Module):
	"""A custom layer that views inputs as ID
	   input.size(0) = batch size """
	def forward(self, input):
		return input.view(input.size(0), -1)


def batchify_data(x_data, y_data, batch_size):
    """Takes a set of data points and labels and groups them into batches
		that can be processed more efficiently than directly runing epoch.
		Only take batch_size chunks (i.e. drop the remainder)
		Refer: Batch Normalization and Dropout in Neural Networks with Pytorch.pdf for more details on batches 
		NOTE:
		torch.tensor is a multi-dimensional matrix containing elements of a single data type.
    """

    N = int(len(x_data) / batch_size) * batch_size
    batches = []
    for i in range(0, N, batch_size):
        batches.append({
            'x': torch.tensor(x_data[i:i+batch_size], dtype=torch.float32),
            'y': torch.tensor(y_data[i:i+batch_size], dtype=torch.long
        )})
    return batches
	

def compute_accuray(predictions, y):
	"""Computes the accuracy of predictions against the gold labels, y
	   NOTE:
	   Moving to numpy will break the graph thus no gradient will be computed so, can't call numpy() on Variable that requires grad. 
	   Gradient is not necessary here so .detach() is explicitly used here.
	"""
	return np.mean(np.equal(predictions.detach().numpy(), y.numpy()))


# ======================== TRAINING PROCEDURES =====================================

def train_model(train_data, dev_data, model, lr=0.01, momentum=0.9, nesterov=False, n_epochs=26):
	"""Train a model for N epochs given data and hyper-parameters. We use SGD here
	NOTE:
	lr => learning rate
	momentum remembers the update dw at each iteration (by accelerating SGD in relevant direction) and determines the next update as linear combination of the gradient and the previous update as
	w = w + dw
	nesterov accelerated gradient calculates the gradient w.r.t. the approximate future position of parameters (rather than w.r.t. our current parameters)
	Refer: gradient descent optimization algorithms.pdf and Stochastic_gradient_descent.pdf for more details
	"""
	train_acc = []
	train_loss = []
	v_acc = []
	v_loss = []

	optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
	for epoch in range(1,n_epochs):
		print("-----------------\nEpoch {}:\n".format(epoch))

		# Run training
		loss, acc = run_epoch(train_data, model.train(), optimizer)
		print("Train loss: {:.6f} | Train accuracy: {:.6f} ".format(loss, acc))
		train_loss.append(loss)
		train_acc.append(acc)

		# Run validation
		val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer)
		print("Validation loss: {:.6f} | Validation accuracy: {:.6f}".format(val_loss, val_acc))
		v_loss.append(val_loss)
		v_acc.append(val_acc)

		# Save model
		# torch.save(model, 'fully_connected_model.pt')
	return val_acc, train_acc, train_loss, v_loss, v_acc
	
def run_epoch(data, model, optimizer):
    """Train model for one pass of train data, and return loss, acccuracy
		NOTE:
		In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
		So, the default action is to accumulate (i.e. sum) the gradients on every loss.backward() call.
		Else the gradient would point in some direction other than the intended direction towards the minimum (or maximum, in case of maximization objectives).
		optimizer.step() then updates the parameter on the current gradient which is stored in .grad attribute of the parameter.
    """
    ###	Gather losses	
    losses = []
    batch_accuracies = []

    ###	If model is in train mode, use optimizer
    is_training = model.training	

    ###	Iterate through batches
    for batch in tqdm(data):
    	x,y = batch['x'],batch['y']				# Grab x and y
    	out = model(x)							# Get output predictions
    	predictions = torch.argmax(out, dim=1)	# Select the max among output predictions tensor as dimension = 1
    	batch_accuracies.append(compute_accuray(predictions, y))	#store accuracy

    	### Comput loss
    	loss = F.cross_entropy(out, y)
    	losses.append(loss.data.item())

    	### If training then update as
    	if is_training:
    		optimizer.zero_grad()
    		loss.backward()
    		optimizer.step()

    ### Calculate epoch level scores
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(batch_accuracies)	
    return avg_loss, avg_accuracy	
