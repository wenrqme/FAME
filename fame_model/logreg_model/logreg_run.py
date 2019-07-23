# Run a saved logistic regression model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

IN_DATAFILE = "../landmarks/upright_landmarks_100_scaled.csv"
IN_COEF = "saved_model/full_coef_v2.txt"
IN_INTERCEPT = "saved_model/full_intercept_v2.txt"

OUT_TXT = "../predictions/logreg_predictions.txt"
OUT_CSV = "../predictions/logreg_predictions.csv"

#load df,  coef_, and intercept_
df = pd.read_csv(IN_DATAFILE, index_col=[0]) #use text.csv for 100 face file, test1000.csv for 1000 face file
coefficients = np.loadtxt(IN_COEF)
intercepts = np.loadtxt(IN_INTERCEPT)
df['gender'] = df['file'].str[3]
print(df.columns)
y = df['gender']
urls = np.array(df.index)
del df['gender']
print('data loaded')

model = LogisticRegression()
coefficients = [coefficients]
model.coef_ = np.array(coefficients)
model.intercept_ = np.array(intercepts)
model.classes_ = np.array([0,1]) #0 for male, 1 for female

all_features = list(df.columns.values)[2:]
landmark_cols = [feat for feat in all_features if (('X_' in feat or 'Y_' in feat) and not 'eye' in feat)]
features = landmark_cols 
X = df[features]
print(X)
print(X.shape)

prediction = model.predict_proba(X)
print("prediction: ", prediction)
score = model.score(X, y)
print("score: ", score)

#np.savetxt(OUT_TXT, prediction) #_full for 1000 faces, _test for 100 faces

# make a dataframe of predictions with filename and save to csv (preferable with url...)
import csv

prediction_format = np.zeros(len(prediction))
for i in range(len(prediction)):
    prediction_format[i] = prediction[i][1] # predicstion gives [probability male, probability female]
#print(prediction_format)

df_predictions = pd.DataFrame({'image_url': urls, 'prediction': prediction_format}, columns=['image_url', 'prediction'])
print(df_predictions)
#df_predictions.to_csv(OUT_CSV)
