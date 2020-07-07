import numpy as np
import sys
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
import copy

trainarg = sys.argv[1]
testarg = sys.argv[2]
data = np.genfromtxt(trainarg, delimiter = ',')
test = np.genfromtxt(testarg, delimiter = ',')

#data = np.genfromtxt('/home/naila/5thSemester/SMAI/assignment2/q4/train.csv',delimiter = ',')
#test = np.genfromtxt('/home/naila/5thSemester/SMAI/assignment2/q4/test.csv',delimiter = ',')

def elastic(trainx,trainy,testx):
	emodel = linear_model.ElasticNet(alpha=0.00006,fit_intercept=True, normalize=True,max_iter=10000)
	emodel.fit(trainx, trainy)
	predicted1 = emodel.predict(testx)
	predicted2 = copy.copy(predicted1)
	threshold = 0.48
	a = 0
	for i in predicted1:
		if i > threshold:
			predicted2[a] = 1
			a = a + 1
		else:
			predicted2[a] = 0
			a = a + 1
#	pelastic = precision_score(testy, predicted2)
#	relastic = recall_score(testy, predicted2)
#	felastic = f1_score(testy,predicted2)
#       accelastic = accuracy_score(testy,predicted2)
	#print((testy,predicted1))
#	print(accelastic,pelastic,relastic,felastic)
	
#	f = open('no3.txt', 'w')
	for i in predicted2:
		print(i)
#	f.close()
	return 1

xtrain = data[:,0:11]
ytrain = data[:,11]
xtest = test[:,0:11]
#ytest = test[:,11]

call1 = elastic(xtrain,ytrain,xtest)
