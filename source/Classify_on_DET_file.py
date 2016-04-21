
# coding: utf-8

# # DET File Structure
# 
# The provided training data include image file raw 16 bits format and the detection list. 
# 
# A detection list associated with the set of images which contains a list of information for the detected objects, in space delimited format with 16 columns and each row representing a single detection in one of the 4 original FITS images. 
# 
# The columns are: 
#     
#    - Unique ID -- An identifier for what detected object a row belongs to
#    
#    
#    - Detection Number -- sequential numbering of detection output of the currently used detection software
#    
#    
#    - Frame Number -- which observation is this row relevant to (1, 2, 3 or 4)
#    
#    
#    - Sexnum -- Source extractor number of the object
#    
#    
#    - Time -- Julian date
#    
#    
#    - RA -- right ascension of object in decimal hours
#    
#    
#    - DEC -- declination in decimal degrees
#    
#    
#    - X -- location in pixels of the object in the original FITS image
#    
#    
#    - Y -- location in pixels of the object in the original FITS image
#    
#    
#    - Magnitude -- brightness of the object in magnitudes
#    
#    
#    - FWHM -- full width at half maximum of Gaussian fit in pixels
#    
#    
#    - Elong -- ratio of long axis to short axis
#    
#    
#    - Theta -- position angle of the long axis
#    
#    
#    - RMSE -- error in fit to straight line
#    
#    
#    - Deltamu -- from Source Extractor, peak value minus threshold over background
#    
#    
#    - Rejected -- this value will be 1 if the operator rejected the detection, 0 otherwise. This column will only be available during the training phase. You need to predict this column
# 

# # TRAINING 
# 
# ## 1. Training with original columns
# 

# In[11]:

import pandas as pd

def onetofour(x):
    return x[0],x[1],x[2],x[3]

names = ["id","det_num", "frame_num", "sex_num", "time", "RA", "DEC", "X", "Y", "mag", "FWHE", "Elong", "theta", "RMSE", "rejected"]
df = pd.read_csv("E:/Dai Hoc/Deep Learning/Near Object/hackspace-2016/data/det_files/01_12DEC03_N01014.det", header=None, names=names, delim_whitespace=True)

dfg = df.groupby('id',as_index=False).agg(lambda x: x.tolist())

new_names = ["RA", "DEC", "X", "Y", "mag", "FWHE", "Elong", "theta", "RMSE", "rejected"]
new_df = pd.DataFrame()

new_df['id'] = dfg['id']

for nn in new_names:
    new_df[nn + '0'],new_df[nn + '1'],new_df[nn + '2'],new_df[nn + '3'] = zip(*dfg[nn].map(onetofour))

# new_df['rejected'] = zip(*dfg['rejected'].map(onebyone))
new_df.drop(['rejected1','rejected2','rejected3'], axis=1, inplace=True)

new_df



# In[4]:

import glob
import os
import pandas as pd

from IPython.display import display, HTML

DIR = "E:/Dai Hoc/Deep Learning/Near Object/hackspace-2016/data/det_files/"
OUT_DIR = "E:/Dai Hoc/Deep Learning/Near Object/hackspace-2016/data/det_training_files/"

names = glob.glob(DIR + "*.det")

header = ["id", "det_num", "frame_num", "sex_num", "time", "RA", "DEC", "X", "Y", "mag", "FWHE", "Elong", "theta", "RMSE", "rejected"]

new_names = ["RA", "DEC", "X", "Y", "mag", "FWHE", "Elong", "theta", "RMSE", "rejected"]

training_input = []

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

for name in names:

    df = pd.read_csv(name, header=None, names=header, delim_whitespace=True)
    
    if df.empty:
        continue
        
    dfg = df.groupby('id',as_index=False).agg(lambda x: x.tolist())
    
    new_df = pd.DataFrame()

    new_df['id'] = dfg['id']

    for nn in new_names:
        new_df[nn + '0'],new_df[nn + '1'],new_df[nn + '2'],new_df[nn + '3'] = zip(*dfg[nn].map(onetofour))

    # new_df['rejected'] = zip(*dfg['rejected'].map(onebyone))
    new_df.drop(['rejected1','rejected2','rejected3'], axis=1, inplace=True)
    
    
    training_input.append(new_df)
    
    pathOutput = OUT_DIR + name[len(DIR):]
    new_df.to_csv(pathOutput, header=None, sep=' ')   
    
    


# In[10]:

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split

training_data = pd.concat(training_input)

Y = training_data['rejected0']
Y = Y.as_matrix()

training_data.drop(['id', 'rejected0'], axis=1, inplace=True)
X = training_data.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

# fit estimator
est = GradientBoostingClassifier(n_estimators=200, max_depth=3)
est.fit(X_train, y_train)

# predict class labels
pred = est.predict(X_test)

# score on test data (accuracy)
acc = est.score(X_test, y_test)
print('ACC: %.4f' % acc)

# predict class probabilities
est.predict_proba(X_test)[0]


    


# In[9]:

from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split

training_data = pd.concat(training_input)


training_data.to_csv("test.csv", header=None, sep=' ')    
Y = training_data['rejected0']
Y = Y.as_matrix()

training_data.drop(['id', 'rejected0'], axis=1, inplace=True)
X = training_data.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print (y_test != y_pred).sum()
print (y_test == y_pred).sum()

