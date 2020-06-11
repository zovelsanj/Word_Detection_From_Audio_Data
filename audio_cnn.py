import numpy as np
# import scipy
# print(np.version.version)
# print(scipy.version.version)
'''
 Checked the version of scipy and numpy to ensure the version compatibility as 
 sklearn.model_selection.train_test_split() was not recognized (FIXED THIS PROBLEM.)
 '''
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import sys
sys.path.append("..")
import utils
from utils import *

# X_train_shape = (2424, 129, 108)
# X_test_shape = (607, 129, 108)
# input_dimension= 129*108

# class MLP(nn.Module):
# 	""" Nothing just Mltilayer Perceptron Model"""
#     def __init__(self, input_dimension):
#         super(MLP, self).__init__()
#         self.flatten = Flatten()
#         # initialize the model layers
#         self.linear1 = nn.Linear(input_dimension,64)    # fully-connected model with a single hidden layer with 64 units.
#         self.linear2 = nn.Linear(64,64)                 # out feature of Linear1 = in feature of Linear2
#         self.linear_out = nn.Linear(64,10)

#     def forward(self, x):
#         xf = self.flatten(x)
#         # use model layers to predict the words
#         out1 = F.relu(self.linear1(xf))
#         out2 = F.relu(self.linear2(out1))
#         out = self.linear_out(out2)
#         return out

def plotConfusionMatrix(predictions, y):
	cm = confusion_matrix(y, predictions.detach().numpy())
	print(cm)
	plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Wistia)
	class_names = ["THE","A","TO","OF","IN","ARE","AND","IS","THAT","THEY"]
	plt.title('confusion_matrix')
	plt.ylabel('True label')
	plt.xlabel('Predicted value')
	points = np.arange(len(class_names))
	plt.xticks(points, class_names, rotation = 45)
	plt.yticks(points, class_names)
	for i in range(10):
		for j in range(10):
			plt.text(j,i, cm[i][j])
	plt.show()		

def plotAccuracy(val_acc, train_acc, train_loss, v_loss, v_acc):
	epoch_num = np.arange(1,30)
	plt.plot(epoch_num, train_acc, label='train')
	plt.plot(epoch_num, v_acc, label='test_validation')
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.title('model accuracy')
	plt.legend()
	plt.tight_layout()
	# plt.grid(True)
	plt.show()
	# plt.savefig('Train and Validation Accuracy.jpeg')	
	
def main():
	### ========= LOAD DATASET =======  
	num_classes = 10
	X_train, X_test, Y_train, Y_test = get_audioData()

	''' Conv2d takes input from 4 dimensional tensor so the 3D tensors i.e. train and test data are reshaped
		Number of input channels is kept one for now '''
	X_train = np.reshape(X_train, (X_train.shape[0], 1,-1, X_train.shape[2]))
	X_test = np.reshape(X_test, (X_test.shape[0], 1, -1, X_test.shape[2]))
	# print(X_train.shape)
	# print(X_test.shape)

	# ======= SPLIT TRAINING SET INTO 10% VALIDATION SET AND 90% TRAIN SET ======
	## REFER: Splitting into train, dev and train sets.pdf and Train_Dev_Test Split.pdf for more information
	dev_split_index = int(9*len(X_train)/10)
	X_dev = X_train[dev_split_index:]
	Y_dev = Y_train[dev_split_index:]
	X_train = X_train[:dev_split_index]
	Y_train = Y_train[:dev_split_index]

	## Shuffling data before training is a good practice for a good Network.
	permutation = np.array([i for i in range(len(X_train))])
	np.random.shuffle(permutation)
	X_train = [X_train[i] for i in permutation]
	Y_train = [Y_train[i] for i in permutation]


	# ======= SPLIT DATASET INTO BATCHES ======
	## REFER: Batch Normalization and Dropout.pdf for more info
	batch_size = 32
	train_batches = batchify_data(X_train,Y_train, batch_size)
	dev_batches = batchify_data(X_dev, Y_dev, batch_size)
	test_batches = batchify_data(X_test, Y_test, batch_size)

	# ========= MODEL SPECIFICATION ========
	""" nn.Sequential is a Module which contains other Modules, and applies them in a sequence. 
		Inputs of Conv2d  input_channel, output_channels, kernel/filter_size(K). Padding(P) = 0 and stride(S) = 1 by default
		Size of input Neuron(I) = input_dimensions of data for first Conv2d(). So, output(O) = [(I-K+2P)+1]
		Output of first Conv2d = input of second Conv2d
		nn.Linear(in_features, out_features). in_features != in_channels and out_features != out_channels. 
		It takes input from 2D tensor and the transformation of 4D tensor for this is done by Flatten.
		Dropout() is for Regularization i.e. avoids problem of overfitting by randomly dropping out few neurons from the network.
		REFER: calculation after conv2d-PyTorchForums.pdf and PyTorch layer dimensions.pdf for more info.
	"""
	model = nn.Sequential(
				nn.Conv2d(1, 64, (3,3)),
				nn.ReLU(),
				nn.MaxPool2d((2,2)),

				nn.Conv2d(64, 128, (3,3)),
				nn.ReLU(),
				nn.MaxPool2d((2,2)),

				Flatten(),
				nn.Linear(96000,128),
				nn.Dropout(0.25),
				nn.Linear(128,64),
				nn.Dropout(0.25),
				nn.Linear(64,10),
			)
	
	# # # =======================================

# 	model = MLP(input_dimension=129*108)
	val_acc, train_acc, train_loss, v_loss, v_acc = train_model(train_batches, dev_batches, model, lr = 0.1, momentum = 0)
	loss, accuracy, predictions, y = run_epoch(test_batches, model.eval(), None)
	print("Loss on test set:" + str(loss) + " Accuracy on test set:" + str(accuracy))
# 	plotAccuracy(val_acc, train_acc, train_loss, v_loss, v_acc)
	plotConfusionMatrix(predictions, y)


if __name__ == '__main__':
	np.random.seed(2321)		## for reproducibility
	torch.manual_seed(2321)
	main()
