import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import seaborn as sns
import pandas as pd
from sklearn.utils import check_array

def mean_absolute_percentage_error(y_true, y_pred): 
    #y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps
		if end_ix > len(sequence)-1:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def create_datasetMultipleTimesBackAhead(dataset, n_steps_out=1, n_steps_in = 1, overlap = 1):
	dataX, dataY = [], []
	tem = n_steps_in + n_steps_out - overlap
	for i in range(int((len(dataset) - tem)/overlap)):
		startx = i*overlap
		endx = startx + n_steps_in
		starty = endx
		endy = endx + n_steps_out
		a = dataset[startx:endx, 0]
		dataX.append(a)
		dataY.append(dataset[starty:endy, 0])
	return np.array(dataX), np.array(dataY)

def PintaResultado(dataset,trainPredict,testPredict,look_back):
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# shift test predictions for plotting
	testPredictPlot = np.empty_like(dataset+1)
	testPredictPlot[:, :] = np.nan
	Ntest = len(testPredict)
	NtestSpace = len(dataset)+1 - (len(trainPredict)+(look_back*2))
	restante = NtestSpace - Ntest
	print(restante)
	testPredictPlot[len(trainPredict)+(look_back*2):len(dataset)+1-restante, :] = testPredict
	#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	#testPredictPlot[len(dataset)-len(testPredict):len(dataset)+1, :] = testPredict
	# plot baseline and predictions
	plt.figure(figsize=(10,4))
	plt.plot(dataset,label='Original Time serie')
	plt.plot(trainPredictPlot,label='Training prediction')
	plt.plot(testPredictPlot,label='Test prediction')
	plt.legend()
	plt.show()


def EstimaRMSE(model,X_train,X_test,y_train,y_test,scaler,look_back):
	# make predictions
	trainPredict = model.predict(X_train.reshape(X_train.shape[0],look_back))
	testPredict = model.predict(X_test.reshape(X_test.shape[0],look_back))
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([y_train.flatten()])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([y_test.flatten()])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	trainScoreMAPE = mean_absolute_percentage_error(trainY[0], trainPredict[:,0])
	testScoreMAPE = mean_absolute_percentage_error(testY[0], testPredict[:,0])
	print('Train Score: %.2f MAPE' % (trainScoreMAPE))
	print('Test Score: %.2f MAPE' % (testScoreMAPE))
	return trainPredict, testPredict

def EstimaRMSE_RNN(model,X_train,X_test,y_train,y_test,scaler,look_back,n_steps):
	# make predictions
	trainPredict = model.predict(X_train.reshape(X_train.shape[0],n_steps,look_back))
	testPredict = model.predict(X_test.reshape(X_test.shape[0],n_steps,look_back))
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([y_train.flatten()])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([y_test.flatten()])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	trainScoreMAPE = mean_absolute_percentage_error(trainY[0], trainPredict[:,0])
	testScoreMAPE = mean_absolute_percentage_error(testY[0], testPredict[:,0])
	print('Train Score: %.2f MAPE' % (trainScoreMAPE))
	print('Test Score: %.2f MAPE' % (testScoreMAPE))
	return trainPredict, testPredict

def EstimaRMSE_MultiStep(model,X_train,X_test,y_train,y_test,scaler,look_back,n_steps):
	# make predictions
	trainPredict = model.predict(X_train.reshape(X_train.shape[0],look_back))
	testPredict = []
	for i in range(X_test.shape[0]):
		temPredict = np.zeros([n_steps])
		for j in range(n_steps):
			if j==0:
				xtest = X_test[i,:]
			else:
				xtest = np.concatenate((X_test[i,j:],temPredict[:j]))
			temPredict[j] = model.predict(xtest.reshape(1,look_back))
		testPredict.append(temPredict)
	testPredict = np.array(testPredict)
	testPredict = testPredict.flatten()
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([y_train.flatten()])
	testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))
	testY = scaler.inverse_transform([y_test.flatten()])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY.reshape(-1, 1), trainPredict.reshape(-1, 1)))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	trainScoreMAPE = mean_absolute_percentage_error(trainY.reshape(-1, 1), trainPredict.reshape(-1, 1))
	testScoreMAPE = mean_absolute_percentage_error(testY[0], testPredict[:,0])
	print('Train Score: %.2f MAPE' % (trainScoreMAPE))
	print('Test Score: %.2f MAPE' % (testScoreMAPE))
	return trainPredict, testPredict

def EstimaRMSE_MultiOuput(model,X_train,X_test,y_train,y_test,scaler,look_back):
	# make predictions
	trainPredict = model.predict(X_train.reshape(X_train.shape[0],look_back))
	testPredict = model.predict(X_test.reshape(X_test.shape[0],look_back))
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict.flatten().reshape(-1, 1))
	trainY = scaler.inverse_transform([y_train.flatten()])
	testPredict = scaler.inverse_transform(testPredict.flatten().reshape(-1, 1))
	testY = scaler.inverse_transform([y_test.flatten()])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	trainScoreMAPE = mean_absolute_percentage_error(trainY[0], trainPredict[:,0])
	testScoreMAPE = mean_absolute_percentage_error(testY[0], testPredict[:,0])
	print('Train Score: %.2f MAPE' % (trainScoreMAPE))
	print('Test Score: %.2f MAPE' % (testScoreMAPE))
	return trainPredict, testPredict

def EstimaRMSE_RNN_MultiStep(model,X_train,X_test,y_train,y_test,scaler,look_back,n_steps,flag):
	# make predictions
	if flag == 1:#multiple times set as features
		trainPredict = model.predict(X_train.reshape(X_train.shape[0],1,look_back))
		testPredict = []
		for i in range(X_test.shape[0]):
			temPredict = np.zeros([n_steps])
			for j in range(n_steps):
				if j==0:
					xtest = X_test[i,:]
				else:
					xtest = np.concatenate((X_test[i,j:],temPredict[:j]))
				temPredict[j] = model.predict(xtest.reshape(1,1,look_back))
			testPredict.append(temPredict)
		testPredict = np.array(testPredict)
		testPredict = testPredict.flatten()
	else: #multiple times set as times
		trainPredict = model.predict(X_train.reshape(X_train.shape[0],look_back,1))
		testPredict = []
		for i in range(X_test.shape[0]):
			temPredict = np.zeros([n_steps])
			for j in range(n_steps):
				if j==0:
					xtest = X_test[i,:]
				else:
					xtest = np.concatenate((X_test[i,j:],temPredict[:j]))
				temPredict[j] = model.predict(xtest.reshape(1,look_back,1))
			testPredict.append(temPredict)
		testPredict = np.array(testPredict)
		testPredict = testPredict.flatten()

	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict.reshape(-1, 1))
	trainY = scaler.inverse_transform(y_train)
	testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))
	testY = scaler.inverse_transform(y_test.flatten().reshape(-1, 1))
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY.reshape(-1, 1), trainPredict.reshape(-1, 1)))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY.reshape(-1, 1), testPredict.reshape(-1, 1)))
	print('Test Score: %.2f RMSE' % (testScore))
	trainScoreMAPE = mean_absolute_percentage_error(trainY.reshape(-1, 1), trainPredict.reshape(-1, 1))
	testScoreMAPE = mean_absolute_percentage_error(testY.reshape(-1, 1), testPredict.reshape(-1, 1))
	print('Train Score: %.2f MAPE' % (trainScoreMAPE))
	print('Test Score: %.2f MAPE' % (testScoreMAPE))
	return trainPredict, testPredict

def EstimaRMSE_RNN_MultiStepEncoDeco(model,X_train,X_test,y_train,y_test,scaler,look_back,n_steps):
	# make predictions
	trainPredict = model.predict(X_train.reshape(X_train.shape[0],look_back,1))
	trainPredict = trainPredict.flatten()
	testPredict = model.predict(X_test.reshape(X_test.shape[0],look_back,1))
	testPredict = testPredict.flatten()
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict.reshape(-1, 1))
	trainY = scaler.inverse_transform(y_train.flatten().reshape(-1, 1))
	testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))
	testY = scaler.inverse_transform(y_test.flatten().reshape(-1, 1))
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY.flatten().reshape(-1, 1), trainPredict.reshape(-1, 1)))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY.flatten().reshape(-1, 1), testPredict.reshape(-1, 1)))
	print('Test Score: %.2f RMSE' % (testScore))
	trainScoreMAPE = mean_absolute_percentage_error(trainY.flatten().reshape(-1, 1), trainPredict.reshape(-1, 1))
	testScoreMAPE = mean_absolute_percentage_error(testY.flatten().reshape(-1, 1), testPredict.reshape(-1, 1))
	print('Train Score: %.2f MAPE' % (trainScoreMAPE))
	print('Test Score: %.2f MAPE' % (testScoreMAPE))
	return trainPredict, testPredict
	
def PlotValidationTimeSeries(datasetO):
	N=np.max(datasetO.index)
	fig, ax = plt.subplots(figsize=(10,7), sharex=True)
	datasetO.plot(0,1,figsize=(10,4), ax=ax)
	ax.axvspan(datasetO.index[0], datasetO.index[int(N*0.6)], color=sns.xkcd_rgb['grey'], alpha=0.5)
	ax.axvspan(datasetO.index[int(N*0.6)], datasetO.index[int(N*0.8)],color=sns.xkcd_rgb['light blue'], alpha=0.5)
	ax.axvspan(datasetO.index[int(N*0.8)], datasetO.index[N],color=sns.xkcd_rgb['light pink'], alpha=0.5)
	plt.text(datasetO.index[int(N*0.2)], 620, "60% Training set")
	plt.text(datasetO.index[int(N*0.62)], 620, "20% Validation set")
	plt.text(datasetO.index[int(N*0.82)], 620, "20% Test set")
	plt.legend().remove()

def PlotCrossvalidationTS():
	n_datasets = 20
	plt.figure(figsize=(10,5))
	for i in range(n_datasets):
		texto = 'Split'+ ' ' + str(i+1)
		plt.text(-2,8-i*0.4,texto)
		plt.arrow(0, 8-i*0.4, 27, 0, alpha=0.3, head_width=0.1, head_length=0.5, fc='k', ec='k')
		m1 = np.arange(1,27)
		y = np.r_[np.zeros(i+6, dtype=int),np.ones(1,dtype=int),2*np.ones(19-i,dtype=int)]
		x2 = [8-i*0.4]*len(m1)
		plt.scatter(m1[:i+6], x2[:i+6] , color='b', alpha=0.5, s=70)
		plt.scatter(m1[i+6],  x2[i+6], color= 'r', alpha=0.5, s=70)
		plt.scatter(m1[i+7:], x2[i+7:], color ='grey', alpha=0.5, s=70)
	plt.text(28,8,'Time')
	plt.axis("off")
	plt.show()

def PlotCrossvalidationTS_Gap():
	n_datasets = 16
	plt.figure(figsize=(10,5))
	for i in range(n_datasets):
		texto = 'Split'+ ' ' +str(i+1)
		plt.text(-2,8-i*0.4,texto)
		plt.arrow(0, 8-i*0.4, 27, 0, alpha=0.3, head_width=0.1, head_length=0.5, fc='k', ec='k')
		m1 = np.arange(1,27)
		y = np.r_[np.zeros(i+6, dtype=int),np.ones(1,dtype=int),2*np.ones(19-i,dtype=int)]
		x2 = [8-i*0.4]*len(m1)
		plt.scatter(m1[:i+6], x2[:i+6] , color='b', alpha=0.5, s=70)
		plt.scatter(m1[i+6:i+10], x2[i+6:i+10], color ='grey', alpha=0.5, s=70)
		plt.scatter(m1[i+10],  x2[i+10], color= 'r', alpha=0.5, s=70)
		plt.scatter(m1[i+11:], x2[i+11:], color ='grey', alpha=0.5, s=70)
	plt.text(28,8,'Time')
	plt.axis("off")
	plt.show()


