"""
This loads a saved linear regression model that was trained on FAME ratings 
and runs it on a set of faces.

inputs:
    landmarks/FAME_landmarks_100.csv
    saved_model/linreg_coef.txt
    saved_model/linreg_intercept.txt
    saved_model/scale_factors_x.txt
    
output:
    predictions/linreg_predictions.txt
    predictions/linreg_predictions.csv

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#---------------------------------
#Global Constants

LANDMARK_FILE = 'landmarks/landmarks_100_scaled.csv'
COEF_FILE = 'saved_model/linreg_coef.txt'
INTERCEPT_FILE = 'saved_model/linreg_intercept.txt'
#SCALE_FACTORS_X = 'saved_model/scale_factors_x.txt'
#SCALE_FACTORS_Y = 'saved_model/scale_factors_y.txt'
scale_y = False

OUT_CSV = "predictions/linreg_predictions.csv"
OUT_TXT = "predictions/linreg_predictions.txt"

#---------------------------------


#load df,  coef_, and intercept_
df = pd.read_csv(LANDMARK_FILE, index_col=[0])
coefficients = np.loadtxt(COEF_FILE)
intercepts = np.loadtxt(INTERCEPT_FILE)
#scale_factors_x = np.loadtxt(SCALE_FACTORS_X)

y = df["rating"]
urls = np.array(df.index)
#scale_mean_x = scale_factors_x[0]
#scale_var_x = scale_factors_x[1]
del df["rating"]
print('data loaded')

model = LinearRegression()
coefficients = [coefficients]
model.coef_ = np.array(coefficients)
model.intercept_ = np.array(intercepts)
model.classes_ = np.array([0,1]) #0 for male, 1 for female

all_features = list(df.columns.values)[2:]
landmark_cols = [feat for feat in all_features if (('X_' in feat or 'Y_' in feat) and not 'eye' in feat)]
features = landmark_cols
X = df[features]
print(X)

#scale
#X -= scale_mean_x
#if all(scale_var_x > 0):
    #X /= scale_var_x
#else:
    #assert 0

if scale_y:
    scale_factors_y = np.loadtxt(SCALE_FACTORS_Y)
    scale_mean_y = scale_factors_y[0]
    scale_var_y = scale_factors_y[1]    
    y -= scale_mean_y
    if scale_var_y > 0:
        y /= scale_var_y
    else:
        assert 0

prediction = model.predict(X)
print("prediction: ", prediction)
score = model.score(X, y)
print("score: ", score)
mse = mean_squared_error(y, prediction)
print("mse: ", mse)
print("root mse: ", mse**0.5)


# make a dataframe of predictions with filename and save to csv (preferable with url...)
import csv

prediction_format = np.zeros(len(prediction))
for i in range(len(prediction)):
    prediction_format[i] = prediction[i][0]
#print(prediction_format)

df_predictions = pd.DataFrame({'image_url': urls, 'prediction': prediction_format}, columns=['image_url', 'prediction'])
print(df_predictions)
df_predictions.to_csv(OUT_CSV)

np.savetxt(OUT_TXT, prediction)
