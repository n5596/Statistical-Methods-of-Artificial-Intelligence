import numpy as np
import math
import sys

trainarg = sys.argv[1]
testarg = sys.argv[2]
readtrain = np.genfromtxt(trainarg, delimiter = ',')
readtest = np.genfromtxt(testarg, delimiter = ',')

train = []
test = []
j = 1
length = len(readtrain)
for x in readtrain:
	if not math.isnan(np.sum(x)):
		train.append(x)
#		j = j+1
#	elif not math.isnan(np.sum(x)) and j > 0.2*length:
#		train.append(x)
#		j = j+1

test = []
for x in readtest:
	if not math.isnan(np.sum(x)):
		test.append(x)


def prediction(data, weights, bias):
	#print(len(weights))
	#print(bias)
	summed = 0
	length = len(data)
#	for i in range(length):
#		summed = summed + data[i]*weights[i]
	summed = np.dot(data, weights)
#	if yvalue == 4:
#		print(summed)
	if summed >= bias:
		return 4
	else:
		return 2

def modifyprediction(data,weights,count):
	summed = 0
	for j in range(len(weights)):
		if np.dot(weights[j],data) >= 0:
			y = 1
		else:
			y = -1
	#	print(y)
		summed = summed + count[j]*y
	if summed >= 0:
		return 4
	else:
		return 2

def estimateweights(traindata, numepochs, bias,eta):
	weights = [0]*(len(traindata[0])-1)
	epochs = 1
	sumerror = 100
	count = 0
	while epochs <= numepochs and sumerror>0:
		sumerror = 0
		flag = 0
		num = len(traindata)
		tp = 0
		fn = 0
		fp = 0
		accuracy = 0
		for x in traindata:
			yvalue = x[10]
			m = np.concatenate(([1],x[1:len(x)-1]), axis=0)
			if yvalue == 4:
				y = 1
			elif yvalue == 2:
				y = -1
			s = y*m
			p = np.dot(s,weights)
			
#			if p > bias:
#				accuracy = accuracy + 1
#			if yvalue == 4 and p > bias:
#				tp = tp + 1
#			if yvalue == 4 and p <= bias:
#				fn = fn + 1
#			if yvalue == 2 and p <= bias:
#				fp = fp + 1

			if p <= bias:
				sumerror = sumerror + 1	
				val = (np.linalg.norm(s))**2
				sc = eta*(bias-p)
				add = sc*(s)/val
               		weights = weights + add
#			print(yvalue, val*eta)
			#print(weights, yvalue, np.dot(s,weights))
#			if p <= bias:
#				estvalue = 4
#			else:
#				estvalue = 2
#			if yvalue != estvalue:
#				weights = weights + eta*(bias-p)*(y*m)/(np.linalg.norm(m))**2
#				sumerror = sumerror + 1
		#print(weights)	

#		accu = float(accuracy)/float(num)
#		if tp == 0 and fn ==0:
#			recall = 0
#		else:
#			recall = float(tp)/float(tp+fn)		
#		if tp == 0 and fp == 0:
#			prec = 0
#		else:	
#			prec = float(tp)/float(tp+fp)	
		
#		print(epochs,sumerror,accu,prec,recall)
		epochs = epochs + 1	
	return weights,sumerror,epochs

def modifyweights(traindata, numepochs):
	w = [0]*(len(traindata[0])-1)
	weights = []
	epochs = 1
	sumerror = 100
	count = []
	c = 0
#	print(len(w))
#	print(traindata)
	while epochs <= numepochs and sumerror>0:
		sumerror = 0
		flag = 0
		accuracy = 0
		tp = 0
		fp = 0
		fn = 0
		num = len(traindata)
		for x in traindata:
			yvalue = x[10]
			m = np.concatenate(([1],x[1:len(x)-1]), axis=0)
			if yvalue == 4:
				y = 1
			elif yvalue == 2:
				y = -1
			u = np.dot(m,w)

			if u*y > 0:
				accuracy = accuracy + 1
#			if yvalue == 4 and u*y > 0:
#				tp = tp + 1
#			if yvalue == 4 and u*y <= 0:
#				fn = fn + 1
#			if yvalue == 2 and u*y <= 0:
#				fp = fp + 1

			if  u*y <= 0:
				sumerror = sumerror + 1	
				weights.append(w)
				w = weights[len(weights)-1] + y*m
				count.append(c)
				c = 0
			else:
				c = c+1
		#print(weights)	

#		accu = float(accuracy)/float(num)
#		if tp == 0 and fn ==0:
#			recall = 0
#		else:
#			recall = float(tp)/float(tp+fn)		
#		if tp == 0 and fp == 0:
#			prec = 0
#		else:	
#			prec = float(tp)/float(tp+fp)				
#		print(epochs,sumerror,accu,prec,recall)
		epochs = epochs + 1	
	return weights,count,sumerror,epochs

def accuracy(testdata, weights,bias,val,c):
	count = 0
	num = len(testdata)
	tp = 0
	fn = 0
	fp = 0
#	print(num)
	for x in testdata:
#		label = x[10]
		m = np.concatenate(([1],x[1:len(x)]), axis=0)
		if val == 1:
			plabel = prediction(m, weights,bias)
		else:
			plabel = modifyprediction(m,weights,c)
		print(plabel)

#		if label == plabel:
#			count = count + 1
#		if plabel == 4 and label == 4:
#			tp = tp + 1
#		if label == 4 and plabel == 2:
#			fn = fn + 1
#		if plabel == 4 and label == 2:
#			fp = fp + 1

	return 	count,tp,fn,fp,num
			
def single(traindata,testdata):
	numepochs1 = 1000
	eta = 0.098
	bias = 75
#	numepochs1 = 1050
#	eta = 0.1
#	bias = 60
	weights1, error1,epochs1 = estimateweights(traindata, numepochs1,bias,eta)
	acc1,tp1,fn1,fp1,num1 = accuracy(testdata,weights1, bias,1,[])

#	a1 = float(acc1)/float(num1)
#	recall1 = float(tp1)/float(tp1+fn1)
#	prec1 = float(tp1)/float(tp1+fp1)

#	print(acc1,tp1,fn1,fp1,num1)
#	print(a1,prec1,recall1)
	return 1

def modified(traindata,testdata):
	numepochs2 = 1493
	weights2, counts2, error2, epochs2 = modifyweights(traindata,numepochs2)
	acc2,tp2,fn2,fp2,num2 = accuracy(testdata,weights2, 0,2,counts2)
	
#	a2 = float(acc2)/float(num2)
#	recall2 = float(tp2)/float(tp2+fn2)
#	prec2 = float(tp2)/float(tp2+fp2)

#	print(a2,prec2,recall2)
	return 1

part1 = single(train,test)
part2 = modified(train,test)
