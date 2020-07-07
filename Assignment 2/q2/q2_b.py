import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense,Dropout,Flatten
from keras.utils import np_utils
from keras import losses
from keras import optimizers
from keras.constraints import maxnorm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from keras.layers.normalization import BatchNormalization
import numpy as np
from sklearn import svm
import sys,os

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def input(file):
    a = unpickle(file)
    data = a["data"]
    labels = a["labels"]
    b = unpickle("batches.meta")
    label_names = b["label_names"]
    return data,labels

#    print data.shape
#    print len(labels)
#    print label_names

trainarg = sys.argv[1]
testarg = sys.argv[2]

os.chdir(trainarg)
a01 = unpickle("data_batch_1")
data1 = a01["data"]

a02 = unpickle("data_batch_2")
data2 = a02["data"]

a03 = unpickle("data_batch_3")
data3 = a03["data"]

a04 = unpickle("data_batch_4")
data4 = a04["data"]


labels1 = a01["labels"] 
labels2 = a02["labels"] 
labels3 = a03["labels"] 
labels4 = a04["labels"] 

b = unpickle("batches.meta")
label_names = b["label_names"]

os.chdir('../')
	
tt = unpickle(testarg)
data5 = tt["data"]
labels5 = tt["labels"] 

data = np.concatenate([data1,data2,data3,data4])
labels = np.concatenate([labels1,labels2,labels3,labels4])

#data = data1
#labels = labels1

x_train = data
y_train = labels
x_test = data5
y_test = labels5

bsize = 128
numclasses = 10
numepochs = 30
ishape = (32,32,3)
ksize = (3,3)
psize = (2,2)
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /=255
x_test /=255

encoder = LabelEncoder()
encoder.fit(y_train)
encoded_ytrain = encoder.transform(y_train)
y_train = np_utils.to_categorical(encoded_ytrain)

encoder.fit(y_test)
encoded_ytest = encoder.transform(y_test)
y_test = np_utils.to_categorical(encoded_ytest)

#y_train = keras.utils.to_categorical(y_train, numclasses)
#y_test = keras.utils.to_categorical(y_test, numclasses)

model = Sequential()
model.add(Conv2D(32, kernel_size=(1, 1), strides=(1, 1),
                 activation='relu',
                 input_shape=ishape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(numclasses, activation='softmax'))

model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,epochs=numepochs,batch_size=bsize)
score = model.evaluate(x_test, y_test,batch_size = bsize)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_classes = model.predict_classes(x_test)

b = unpickle("batches.meta")
label_names = b["label_names"]
#print(b)
f = open('q2_b_output.txt', 'w')
for i in y_classes:
	l = label_names[i]
	print >> f,(l)
f.close()
