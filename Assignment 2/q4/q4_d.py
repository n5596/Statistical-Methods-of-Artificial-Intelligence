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

def ordinary(trainx,trainy,testx):
	omodel = linear_model.LinearRegression(fit_intercept=True, normalize=False)
	omodel.fit(trainx, trainy)
	predicted1 = omodel.predict(testx)
	predicted2 = copy.copy(predicted1)
	threshold = 0.56
	a = 0
	for i in predicted1:
		if i > threshold:
			predicted2[a] = 1
			a = a + 1
		else:
			predicted2[a] = 0
			a = a + 1
#	pordinary = precision_score(testy, predicted2)
#	rordinary = recall_score(testy, predicted2)
#	fordinary = f1_score(testy, predicted2)
#        acco = accuracy_score(testy,predicted2)
	#print((testy,predicted1))
#	print(acco,pordinary,rordinary,fordinary)

#	f = open('no4.txt', 'w')
	for i in predicted2:
		print(i)
#	f.close()
	return 1


xtrain = data[:,0:11]
ytrain = data[:,11]
xtest = test[:,0:11]
#ytest = test[:,11]

call1 = ordinary(xtrain,ytrain,xtest)
