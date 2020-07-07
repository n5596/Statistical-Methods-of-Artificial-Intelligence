import numpy as np
import sys
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
import copy

#trainarg = sys.argv[1]
#testarg = sys.argv[2]
#data = np.genfromtxt(trainarg, delimiter = ',')
#test = np.genfromtxt(testarg, delimiter = ',')

data = np.genfromtxt('/home/naila/5thSemester/SMAI/assignment2/q3/notMNIST_train_data.csv',delimiter = ',')
datalabels = np.genfromtxt('/home/naila/5thSemester/SMAI/assignment2/q3/notMNIST_train_labels.csv',delimiter = ',')
test = np.genfromtxt('/home/naila/5thSemester/SMAI/assignment2/q3/notMNIST_test_data.csv',delimiter = ',')
testlabels = np.genfromtxt('/home/naila/5thSemester/SMAI/assignment2/q3/notMNIST_test_labels.csv',delimiter = ',')

def logistic(trainx,trainy,testx,testy):
	values = np.array([0.001,0.5,1,1.25,1.5,10])
	lambdas = np.array([1000,2,1,0.8,0.666,0.1])
	c1 = []
	c2 = []
	n1 = []
	n2 = []
	for c in values:
		l2model = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=c, fit_intercept=True)
		l2model.fit(trainx, trainy)
		predicted2 = l2model.predict(testx)
		weights2 = l2model.coef_
		p2logistic = precision_score(testy, predicted2)
		r2logistic = recall_score(testy, predicted2)
		f2logistic = f1_score(testy, predicted2)
#	print(weights2)
		w2 = weights2.ravel()
		c2.append(l2model.coef_)
#	w2 = w2[1:len(w2)]
#		plt.imshow(np.abs(w2.reshape(28,28)), interpolation='nearest',cmap='binary', vmax=None, vmin=None)
#		plt.show()
	#print((testy,predicted2))
#		print(accuracy_score(testy,predicted2),p2logistic,r2logistic,f2logistic)

		l1model = linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=c, fit_intercept=True)
		l1model.fit(trainx, trainy)
		predicted1 = l1model.predict(testx)
		weights1 = l1model.coef_
		p1logistic = precision_score(testy, predicted1)
		r1logistic = recall_score(testy, predicted1)
		f1logistic = f1_score(testy, predicted1)
#	print(weights1)
		w1 = weights1.ravel()
		c1.append(l1model.coef_)
#	w1 = w1[1:len(w1)]
#		plt.imshow(np.abs(w1.reshape(28,28)), interpolation='nearest',cmap='binary', vmax=None, vmin=None)
#		plt.show()
	#print((testy,predicted1))
#		print(accuracy_score(testy,predicted1),p1logistic,r1logistic,f1logistic)
		norm1 = np.linalg.norm(weights1)
		norm2 = np.linalg.norm(weights2)
		n1.append(norm1)
		n2.append(norm2)
#		print(norm1, norm2)

#	ax = plt.gca()
#	ax.plot(lambdas, n1)
#	plt.xlabel('lambdas')
#	plt.ylabel('norm1')
#	plt.title('Norm of weight vectors as a function of lambda in L1')
#	plt.axis('tight')
#	plt.show()
	
#	ax = plt.gca()
#	ax.plot(lambdas, n2)
#	plt.xlabel('lambdas')
#	plt.ylabel('norm2')
#	plt.title('Norm of weight vectors as a function of lambda in L2')
#	plt.axis('tight')
#	plt.show()
	return 1

#s = data.shape
#s1 = s[0]
#s2 = s[1]
#m1 = np.ones(s1)
#datax = np.concatenate(((m1)[:, np.newaxis], data), axis=1)

#t = test.shape
#t1 = t[0]
#m2 = np.ones(t1)
#datay = np.concatenate(((m2)[:, np.newaxis], test), axis=1)

datax = copy.copy(data)
datay = copy.copy(test)

call1 = logistic(datax,datalabels,datay,testlabels)
