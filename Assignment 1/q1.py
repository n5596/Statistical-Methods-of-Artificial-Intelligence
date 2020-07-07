import numpy as np
import sys

trainarg = sys.argv[1]
testarg = sys.argv[2]
data = np.genfromtxt(trainarg, delimiter = ',')
test = np.genfromtxt(testarg, delimiter = ',')
#print(len(data))
#print(data)

def prediction(data, weights, bias):
	#print(len(weights))
	#print(bias)
	summed = 0
	data[0] = 1
	length = len(data)
	summed = np.dot(data,weights)
#	for i in range(length):
#		summed = summed + data[i]*weights[i]
	#print(summed)
	if summed >= bias:
		return 1
	else:
		return 0

def estimate(traindata, numepochs, bias, sval):
	weights = [0]*(len(traindata[0]))
	epochs = 1
	#print(len(traindata))
	flag = 0
	sumerror = 100
	while epochs <= numepochs and sumerror>0:
		sumerror = 0
		flag = 0
		s = []
		accuracy = 0
		tp = 0
		fn = 0
		fp = 0
		num = len(traindata)
		for x in traindata:
			j = 0
			yvalue = x[0]
			if yvalue == 1:
				y = 1
			else:
				y = -1
			m = np.concatenate(([1],x[1:len(x)]), axis=0)
			#print(y)
			estvalue = prediction(m, weights,bias)
			error = (yvalue - estvalue)**2
			sumerror = sumerror + error
			if estvalue == yvalue:
				accuracy = accuracy + 1
			if estvalue == 1 and yvalue == 1:
				tp = tp + 1
			if yvalue == 1 and estvalue == 0:
				fn = fn + 1 
			if estvalue == 1 and yvalue == 0:
				fp = fp + 1
			#print(sumerror)
			if estvalue != yvalue and sval == 0:
#				for i in range(len(m)):
#					weights[i] = weights[i] + y*m[i]
				weights = weights + y*m
			if estvalue != yvalue and sval == 1:
				s.insert(j,x)
				#print(s)
				j = j + 1
		#	if flag == 0:
		#		print(weights[500:530])	
		#		flag = 1
		accu = float(accuracy)/float(num)
		if tp == 0 and fn ==0:
			recall = 0
		else:
			recall = float(tp)/float(tp+fn)		
		if tp == 0 and fp == 0:
			prec = 0
		else:	
			prec = float(tp)/float(tp+fp)		
		#print(s)
		if sval == 1:	
			for x in s:
				if x[0] == 1:
					l = 1
				else:
					l = -1
				m = np.concatenate(([1],x[1:len(x)]), axis=0)
#				for i in range(len(m)):
#					weights[i] = weights[i] + l*m[i]
				weights = weights + l*m
		#print(weights[500:530])				
#		print(epochs,sumerror,accu,prec,recall)	
		epochs = epochs + 1
	return weights,sumerror,epochs
	#return 1,1,1

def accuracy(test, weights,bias):
	count = 0
	num = len(test)
#	print(num)
	tn = 0
	fn = 0
	tp = 0 
	fp = 0
	for x in test:
#		label = x[0]
		m = np.concatenate(([1],x[0:len(x)]), axis=0)
		plabel = prediction(m, weights,bias)
		print(plabel)
#		if label == plabel:
#			count = count + 1
#		if plabel == 1 and label == 1:
#			tp = tp + 1
#		if label == 1 and plabel == 0:
#			fn = fn + 1
#		if plabel == 1 and label == 0:
#			fp = fp + 1
#	ans = count/num
	return 	count,tp,fn,fp,num
			
def single(traindata,testdata):
	numepochs1 = 100
#	print("one")
	#print(testdata[17])
	weights1, error1,epochs1 = estimate(traindata, numepochs1,0,0)
	acc1,tp1,fn1,fp1,num1 = accuracy(testdata,weights1,0)
	#print(testdata[17])
	#precision = tp/(tp+fp)
#	recall1 = float(tp1)/float(tp1+fn1)
#	prec1 = float(tp1)/float(tp1+fp1)
#	a1 = float(acc1)/float(num1)
#	print(a1,prec1,recall1)
	#print("%f %f"%(precision,recall))
	return 1

def singlemargin(traindata, testdata, margin2):
	numepochs2 = 100
#	print("two")
	#print(testdata[17])
	weights2, error2, epochs2 = estimate(traindata, numepochs2, margin2,0)
	acc2, tp2, fn2,fp2, num2 = accuracy(testdata,weights2,margin2)
	#print(testdata[17])
	#precision = tp/(tp+fp)
#	recall2 = float(tp2)/float(tp2+fn2)
#	a2 = float(acc2)/float(num2)
#	prec2 = float(tp2)/float(tp2+fp2)
#	print(a2,prec2,recall2)
	#print("%f %f"%(precision,recall))
	return 1

def batch(traindata, testdata):
	numepochs3 = 600
	weights3, error3, epochs3 = estimate(traindata, numepochs3,0,1)
	acc3,tp3,fn3,fp3,num3 = accuracy(testdata,weights3,0)
	#	precision = tp/(tp+fp)
#	recall3 = float(tp3)/float(tp3+fn3)
#	prec3 = float(tp3)/float(tp3+fp3)
	#	print("%f %f"%(precision,recall))
#	a3 = float(acc3)/float(num3)
#	print(a3,prec3,recall3)
	return 1

def batchmargin(traindata, testdata, margin4):
	numepochs4 = 600
	weights4, error4, epochs4 = estimate(traindata, numepochs4, margin4,1)
	acc4,tp4,fn4,fp4,num4 = accuracy(testdata, weights4, margin4)
	#	print(acc4)
	#	precision = tp/(tp+fp)
#	prec4 = float(tp4)/float(tp4+fp4)
#	recall4 = float(tp4)/float(tp4+fn4)
#	a4 = float(acc4)/float(num4)
#	print(a4,prec4,recall4)
	#	print("%f %f"%(precision,recall))
	return 1

part1 = single(data,test)
part2 = singlemargin(data,test,1.8)
part3 = batch(data,test)
part4 = batchmargin(data,test,1.8)
