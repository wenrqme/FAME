"""
This trains a logistic regression model to predict binary gender

inputs:
    "../landmarks/a_scaled.csv"

outputs:
    "full_coef_new.txt"
    "full_intercept_new.txt"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from scipy.stats import ttest_ind

IN_DATAFILE = "../landmarks/a_scaled.csv"
OUT_COEF = "saved_model/full_coef_new.txt"
OUT_INTERCEPT = "saved_model/logreg/full_intercept_new.txt"

df = pd.read_csv(IN_DATAFILE)
#df = pd.read_csv('UTKLandmarks/test.csv')
print('data loaded')

#remove 100 samples that will be used to test it later
filelist = pd.read_csv('../../mturk_surveys/survey1_mflikert/clean_results.csv')
filelist = filelist['Image_filename']
filelist = list(filelist)
df['filename'] = df['filename'].str[8:]
df_exclude = df[df['filename'].isin(filelist)]
df = pd.concat([df, df_exclude]).drop_duplicates(keep=False)

# df_male = df[df['gender'] == 1].copy()
# df_female = df[df['gender'] == 0].copy()
#print(df_exclude)
#print(df)

features = list(df.columns.values)[2:]
ALL_FEAT = features
key_landmark_cols = ['x_0', 'x_4', 'x_11', 'x_14', 'y_8', 'y_57', 'y_51', 'y_29'] # not decided
target = 'gender'

print('num features: ', len(features))
print(features)
#print(df)

from sklearn.linear_model import LogisticRegression

niter = 100
best_model = None
best_accuracy = 0
saved_loss = np.inf

ave_test_logloss = []
ave_test_score = []
ave_train_logloss = []
ave_train_score = []
# features = key_landmark_cols # use later
for i in range(niter):
    print('iter # ', i)
    C = [0.03,.1,.3,1,3,10,30,100]
    traindev_x, test_x, traindev_y, test_y = \
        train_test_split(df[features], df[target], train_size=.80)
    train_x, dev_x, train_y, dev_y = \
        train_test_split(traindev_x, traindev_y, train_size=.75)

    # pick best C
    min_loss = np.inf
    best_c = None
    for c in C:
        logReg = LogisticRegression(C = c, intercept_scaling=1e6, penalty='l1')
        logReg.fit(train_x, train_y)
        loss = log_loss(dev_y, logReg.predict_proba(dev_x))
        #print(loss)
        if loss <= min_loss:
            min_loss = loss
            best_c = c

    # run with best C
    logReg2 = LogisticRegression(C=best_c, 
                                 intercept_scaling=1e6, penalty='l1')  
    logReg2.fit(traindev_x, traindev_y)

    test_logloss = log_loss(test_y, logReg2.predict_proba(test_x))
    test_score = logReg2.score(test_x, test_y)
    train_logloss = log_loss(traindev_y, logReg2.predict_proba(traindev_x))
    train_score = logReg2.score(traindev_x, traindev_y)

    ave_test_logloss.append(test_logloss)
    ave_test_score.append(test_score)
    ave_train_logloss.append(train_logloss)
    ave_train_score.append(train_score)
    print('\nLog Loss: ', test_logloss)
    print('\nTest accuracy: ', test_score)
    if test_score > best_accuracy:
        best_accuracy = test_score
        saved_loss = test_logloss
        best_model = logReg2
    #print('goodbye')

# convert to numPy arrays
ave_test_logloss = np.array(ave_test_logloss)
ave_test_score = np.array(ave_test_score)
ave_train_logloss = np.array(ave_train_logloss)
ave_train_score = np.array(ave_train_score)
    
print('\nAve Test Log Loss: ', ave_test_logloss.mean())
print('\nAve Test accuracy: ', ave_test_score.mean())

print('\nAve Train Log Loss: ', ave_train_logloss.mean())
print('\nAve Train accuracy: ', ave_train_score.mean())

print('\nBest Test Accuracy: ', best_accuracy)
print('\nLog Loss: ', saved_loss)
print('\nweights: ', best_model.coef_)
print('\nintercept: ', best_model.intercept_)

#save weights & decision function to file
np.savetxt(OUT_COEF, best_model.coef_)
np.savetxt(OUT_INTERCEPT, best_model.intercept_)

print('goodbye')
