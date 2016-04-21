from PIL import Image
import numpy as np


def loadImage(data):
    img = Image.open(data, 'r')
    res = []
    for i in xrange(4):
        part = img.crop((0+74*i,0,64+74*i,64))
        arr = np.fromstring(part.tobytes(), dtype='uint8', count=-1, sep='').reshape(part.size + (len(part.getbands()),))
        res.append(arr)
    return np.asanyarray(res, dtype='uint8')

import os
data = []
# f = open('workfile', 'w')
f = open(r'/home/tenma/Desktop/hackspace-2016/data/det_files/'+'01_12DEC04_N26006'+'.det')
# print f
txtLines = f.readlines()
tempTarget = txtLines[0].split(' ')
print tempTarget[len(tempTarget)-1]
target = []
for file in os.listdir(r'/home/tenma/Desktop/hackspace-2016/data/images'):
    f = open(r'/home/tenma/Desktop/hackspace-2016/data/det_files/'+file+'.det')
    txtLines = f.readlines()
    if(len(txtLines) != len(os.listdir(r'/home/tenma/Desktop/hackspace-2016/data/images/'+file))*4):
        print 'Wrong'
        break
    for i in xrange(len(txtLines)/4):
        tempTarget = txtLines[i*4].split(' ')
        target.append(int(tempTarget[len(tempTarget)-1]))
    for files in os.listdir(r'/home/tenma/Desktop/hackspace-2016/data/images/'+file):
        data.append(loadImage(r'/home/tenma/Desktop/hackspace-2016/data/images/'+file+'/'+files))
        

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.3, random_state = 42)        

import pickle
pickle.dump(((X_train,y_train),(X_test, y_test), (X_test,y_test)), open('data.pkl', 'w'))