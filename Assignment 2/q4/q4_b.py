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

def ridge(trainx,trainy,testx):
	rmodel = linear_model.Ridge(alpha=0.1, fit_intercept=True, normalize=True,max_iter=10000, tol=0.001)
	rmodel.fit(trainx, trainy)
	predicted1 = rmodel.predict(testx)
	predicted2 = copy.copy(predicted1)
	threshold = 0.425
	a = 0
	for i in predicted1:
		if i > threshold:
			predicted2[a] = 1
			a = a + 1
		else:
			predicted2[a] = 0
			a = a + 1
#	pridge = precision_score(testy, predicted2)
#	rridge = recall_score(testy, predicted2)
#	fridge = f1_score(testy,predicted2)
#       accridge = accuracy_score(testy,predicted2)
	#print((testy,predicted1))
#	print(accridge,pridge,rridge,fridge)

#	f = open('no2.txt', 'w')
	for i in predicted2:
		print(i)
#	f.close()
	return 1

xtrain = data[:,0:11]
ytrain = data[:,11]
xtest = test[:,0:11]
#ytest = test[:,11]

call1 = ridge(xtrain,ytrain,xtest)
