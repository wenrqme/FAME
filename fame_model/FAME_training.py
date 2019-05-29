"""
This will train an L1 linear regression model (aka LASSO) using the 1000 mturk
labels (i.e. FAST ratings). The features used are: all 136 landmark features 
The model is then written to disk for later use.
10 fold cross validation is used to estimate performance. The fold with
the highest accuracy is the one that is written.
Hyper parameter selection is done using a dev set.

inputs:
    landmarks/FAME_landmarks.csv
    
output:
    linreg_coef.txt
    linreg_intercept.txt
    scale_factors_x.txt

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Lasso

#------------------------
# GLOBAL CONST

DATAFILE = 'landmarks/landmarks_merged.csv'
niter = 10

# we will use sklearn LASSO model which has a hyhperparameter: alpha 
# alpha: Constant that multiplies the L1 term. Defaults to 1.0. 
# alpha = 0 is equivalent to an ordinary least squares
alphas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 3,10,100,1000]


#------------------------------


df = pd.read_csv(DATAFILE) # replace with file with human labeled data
print('data loaded')

features = list(df.columns.values)[2:]
target = "FAST_rating"

best_model = None
best_mse = np.inf
root_mse = np.inf

ave_test_mse = []
ave_test_score = []
ave_train_mse = []
ave_train_score = []
ave_test_rootmse = []
ave_train_rootmse = []


x = df[features]

#scale(x)
mean = x.mean()
var = x.var()
x -= mean
if all(var > 0):
    x /= var
else:
    assert 0
    
y = df[target]

#scale y
#y_mean = y.mean()
#y_var = y.var()
#y -= y_mean
#if y_var > 0:
    #y /= y_var
#else:
    #assert 0

# repeat KFold multiple times to get an even better estimate
for i in range(niter):
    print('iter # ', i)
    
    #using 10 fold cross validation
    kf = KFold(n_splits=10, shuffle = True)
    
    for train_index, test_index in kf.split(x):
        traindev_x, test_x = x.iloc[train_index], x.iloc[test_index]
        traindev_y, test_y = y.iloc[train_index], y.iloc[test_index]
        train_x, dev_x, train_y, dev_y = \
            train_test_split(traindev_x, traindev_y, train_size=.5)    
    
        # pick best alpha
        min_mse = np.inf
        best_a = None
        for a in alphas:
            lasso = Lasso(alpha = a, max_iter=10000, tol=.0001)
            lasso.fit(train_x, train_y)
            
            #calculate mean squared error
            mse = mean_squared_error(dev_y, lasso.predict(dev_x))
            #print(mse)
            if mse <= min_mse:
                min_mse = mse
                best_a = a
    
        # run with best C
        print(best_a)
        lasso2 = Lasso(alpha = best_a, max_iter=10000, tol=.0001)
        lasso2.fit(traindev_x, traindev_y)
    
        test_mse = mean_squared_error(test_y, lasso2.predict(test_x))
        test_score = lasso2.score(test_x, test_y)
        train_mse = mean_squared_error(traindev_y, lasso2.predict(traindev_x))
        train_score = lasso2.score(traindev_x, traindev_y)
        test_rootmse = test_mse**0.5
        train_rootmse = train_mse**0.5
    
        ave_test_mse.append(test_mse)
        ave_test_score.append(test_score)
        ave_train_mse.append(train_mse)
        ave_train_score.append(train_score)
        ave_test_rootmse.append(test_rootmse)
        ave_train_rootmse.append(train_rootmse)
        
        print('\nMSE: ', test_mse)
        print('Test score: ', test_score)
        
        if test_mse < best_mse:
            best_mse = test_mse
            root_mse = test_rootmse
            best_model = lasso2
        # TODO, also add a root mse


# convert to numPy arrays
ave_test_score = np.array(ave_test_score)
ave_train_score = np.array(ave_train_score)
ave_test_mse = np.array(ave_test_mse)
ave_train_mse = np.array(ave_train_mse)
ave_test_rootmse = np.array(ave_test_rootmse)
ave_train_rootmse = np.array(ave_train_rootmse)

# summary
print('\nAve Test score: ', ave_test_score.mean())
print('Ave Test MSE: ', ave_test_mse.mean())
print('Ave Test root MSE: ', ave_test_rootmse.mean())
print('\nAve Train score: ', ave_train_score.mean())
print('Ave Train MSE: ', ave_train_mse.mean())
print('Ave Train root MSE:', ave_train_rootmse.mean())

print('\n\nBest Test root MSE: ', root_mse)
print('weights: ', best_model.coef_)
print('intercept: ', best_model.intercept_)

x_scale_factors = np.array([mean, var])
#y_scale_factors = np.array([y_mean, y_var])
np.savetxt('saved_model/scale_factors_x1.txt', x_scale_factors)
#np.savetxt('scale_factors_y.txt', y_scale_factors)
np.savetxt("saved_model/linreg_coef1.txt", best_model.coef_)
np.savetxt("saved_model/linreg_intercept1.txt", np.array([best_model.intercept_]))

print('goodbye')
