from PIL import Image
import numpy as np
import theano

size = 28,28

def loadImage(data):
    img = Image.open(data, 'r').convert('L')
    res = []
    for i in xrange(4):
        part = img.crop((0+74*i,0,64+74*i,64))
        part.thumbnail(size, Image.ANTIALIAS)
        #arr = np.fromstring(part.tobytes(), dtype='uint8', count=-1, sep='').reshape(part.size + (len(part.getbands()),))
        dataNor = part.getdata()
        for m in xrange(len(dataNor)):
            for n in xrange(len(dataNor[0])):
                dataNor /= 255.0
        arr = np.array(dataNor,
                    'float32')
        # print arr.dtype
        res.extend(arr)
    return np.asanyarray(res, dtype='float32')
    # theano.config.floatX

import os
data = []
# f = open('workfile', 'w')
# f = open(r'D:/hackathons/nasa/hackspace-2016/data/det_files/'+'01_12DEC04_N26006'+'.det')
# print f
# txtLines = f.readlines()
# tempTarget = txtLines[0].split(' ')
# print tempTarget[len(tempTarget)-1]
target = []
count = 0
numOfPos = 0
numOfNev = 0
for file in os.listdir(r'D:/hackathons/nasa/hackspace-2016/data/images'):
    # if(len(target) > 100):
    #     print 'Break'
    #     break
    f = open(r'D:/hackathons/nasa/hackspace-2016/data/det_files/'+file+'.det')
    txtLines = f.readlines()
    if(len(txtLines) != len(os.listdir(r'D:/hackathons/nasa/hackspace-2016/data/images/'+file))*4):
        print 'Wrong'
        continue
    for i in xrange(len(txtLines)/4):
        tempTarget = txtLines[i*4].split(' ')
        target.append(int(tempTarget[len(tempTarget)-1]))

    for files in os.listdir(r'D:/hackathons/nasa/hackspace-2016/data/images/'+file):
        count += 1
        data.append(loadImage(r'D:/hackathons/nasa/hackspace-2016/data/images/'+file+'/'+files))

print ('Num of sample: ', count)
# dataT = np.array(data)
# print (dataT.shape)
for i in xrange(len(target)):
    if(target[i] == 1):
        numOfPos += 1
    if(target[i] == 0):
        numOfNev += 1
print ('numOfPos: ', numOfPos)
print ('numOfNev: ', numOfNev)

targetT = np.array(target)

from sklearn.cross_validation import train_test_split
X_train, X_test_valid, y_train, y_test_valid = train_test_split(data, targetT, test_size = 0.4, random_state = 42)

X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid, y_test_valid, test_size = 0.5, random_state = 42)

# X_train = np.array(X_trainD)
# X_test = np.array(X_testD)

# print (X_train.dtype)
# print (y_train.dtype)

# print (X_train[0])
X_trainFinal = np.array(X_train)
X_testFinal = np.array(X_test)
X_validFinal = np.array(X_valid)
import pickle
pickle.dump(((X_trainFinal,y_train),(X_validFinal, y_valid), (X_testFinal,y_test)), open('data.pkl', 'w'))