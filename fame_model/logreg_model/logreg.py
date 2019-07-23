"""
combines all py files for running the logistic regression model
This is only for retraining and testing the logistic regression model, and does not label the deception data

"""

import glob
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from scipy.stats import ttest_ind

COEF = "saved_model/full_coef_v2.txt"
INTERCEPT = "saved_model/full_intercept_v2.txt"


def main():
    # function calls to train FAME and test it on the 100 face test set
    #landmarks = prepare_data("F:/processed/", "landmarks/openface_14000.csv")
    #landmarks = "../landmarks/c.csv"
    #landmarks_scaled = scale_landmarks(landmarks, "../landmarks/c_scaled.csv")
    #landmarks_scaled_upright = scan_upright_files(landmarks_scaled, "../landmarks/c_scaled_upright.csv")
    #coef, intercept = train(landmarks_scaled_upright)
    #predictions = run_model("../landmarks/upright_landmarks_100_scaled.csv", coef, intercept, "predictions/linreg_predictions.csv")
    predictions = run_model("../landmarks/upright_landmarks_100_scaled.csv", COEF, INTERCEPT, "../predictions/logreg_predictions.csv")
    
    
def prepare_data(openface_folder, output_landmark_file):
    list_ = []
    i = 1;
    
    for file in glob.glob(openface_folder + "*.csv"):
        df = pd.read_csv(file, index_col=None, header=0)
        filename = os.path.splitext(file)[0][23:]
        print(filename)
        try:
            df.insert(0, 'gender', filename[3])
            df.insert(0, 'rating', df_ratings.loc[filename, 'Answer.Rating'])
            df.insert(0, 'file', filename)
            list_.append(df)
        except KeyError:
            print("not in mturk file")
        print(i)
        i += 1
        
    landmarks = pd.concat(list_, axis = 0, ignore_index = True)
    print(len(landmarks))
    
    landmarks.to_csv(output_landmark_file, encoding='utf-8',index=False)
    return output_landmark_file  


# scales the landmarks from openface, centering the face around the nose,
# and using the distance between the eyes as a reference for the size of the face.
def scale_landmarks(landmark_file, scaled_landmark_file):
    df = pd.read_csv(landmark_file, skipinitialspace=True)
    for i in df.columns:
        print(i)    
    new_columns = df.columns.values
    new_columns[0] = 'file'
    df.columns = new_columns
    df = df.set_index('file')
    
    print(df.columns)
    
    x_cols = [col for col in df.columns if 'X_' in col]
    y_cols = [col for col in df.columns if 'Y_' in col]
    all_cols = [col for col in df.columns if (('X_' in col or 'Y_' in col) and not 'eye' in col)]
    print(all_cols)
    
    for i in df.index:
        print(i)
        df.loc[i, x_cols] -= df.loc[i, 'X_30']
        df.loc[i, y_cols] -= df.loc[i, 'Y_30']
        
        difference = df.loc[i, 'X_42'] - df.loc[i, 'X_39']
        df.loc[i, all_cols] /= difference
        
    print(df)
    df.to_csv(scaled_landmark_file)
    return scaled_landmark_file

#scan for upright face files
def scan_upright_files(in_openface_landmarks, out_landmarks):
    df = pd.read_csv(in_openface_landmarks, skipinitialspace=True)
    #print(df.head())
    grouped = df.groupby(['file'])
    #display(grouped.first())
    #print(type(grouped.get_group('2018-02-17_14-33-35-477-I-T-kingferry_openface.csv')))
    #print(type(grouped))
    
    CONF_THRESH = 0.90
    POSE_RX_THRESH = 10*np.pi/180
    POSE_RY_THRESH = 10*np.pi/180
    POSE_RZ_THRESH = 3*np.pi/180
    
    landmarks = pd.DataFrame(columns=df.columns)
    failed_files = []
    count = 0
    total = 0
    
    #scan through the openface values
    for name, group in grouped:
        df = grouped.get_group(name)
        total += 1
        
        count_for_file = 0
        for index, row in df.iterrows():
            if (row['confidence'] > CONF_THRESH) and  (np.absolute(row['pose_Rx']) < POSE_RX_THRESH) and (np.absolute(row['pose_Ry']) < POSE_RY_THRESH) and (np.absolute(row['pose_Rz']) < POSE_RZ_THRESH):
                print('index: ', index)
                print(index, name)
                landmarks.loc[count] = row #is this possible?
                count += 1
                count_for_file += 1
                break
            
        if count_for_file == 0:
            print('failed: ', name)
            failed_files.append(name)
    
    failed_files = np.array(failed_files)
    
    print('count: ', count)
    print('total: ', total)
    #print(failed_files)
    
    landmarks.to_csv(out_landmarks)
    return out_landmarks


def train(IN_DATAFILE):
    #OUT_COEF = "saved_model/full_coef_v2.txt"
    #OUT_INTERCEPT = "saved_model/full_intercept_v2.txt"
    
    df = pd.read_csv(IN_DATAFILE)
    #df = pd.read_csv('UTKLandmarks/test.csv')
    print('data loaded')
    
    #remove 100 samples that will be used to test it later
    filelist = pd.read_csv('../../mturk_surveys/survey1_mflikert/clean_results.csv')
    filelist = filelist['Image_filename']
    filelist = list(filelist)
    df_exclude = df[df['file'].isin(filelist)]
    df = pd.concat([df, df_exclude]).drop_duplicates(keep=False)
    
    # df_male = df[df['gender'] == 1].copy()
    # df_female = df[df['gender'] == 0].copy()
    #print(df_exclude)
    #print(df)
    
    all_features = list(df.columns.values)[2:]
    landmark_cols = [feat for feat in all_features if (('X_' in feat or 'Y_' in feat) and not 'eye' in feat)]
    features = landmark_cols
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
    np.savetxt(COEF, best_model.coef_)
    np.savetxt(INTERCEPT, best_model.intercept_)
    
    return COEF, INTERCEPT

def run_model(landmarks_file, coef_file, intercept_file, out_predictions):
    #load df,  coef_, and intercept_
    df = pd.read_csv(landmarks_file, index_col=[0]) #use text.csv for 100 face file, test1000.csv for 1000 face file
    coefficients = np.loadtxt(coef_file)
    intercepts = np.loadtxt(intercept_file)
    df['gender'] = df['file'].str[3].astype(int)
    y = df['gender']
    urls = np.array(df['file'])
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
    print('\n')
    print(X)
    print(y)
    print(X.shape)
    print(y.shape)
    
    prediction = model.predict_proba(X)
    print("\nprediction: ", prediction)
    score = model.score(X, y)
    print("score: ", score)
    
    prediction_format = np.zeros(len(prediction))
    for i in range(len(prediction)):
        prediction_format[i] = prediction[i][1] # predicstion gives [probability male, probability female]
    #print(prediction_format)
    
    df_predictions = pd.DataFrame({'image_url': urls, 'prediction': prediction_format}, columns=['image_url', 'prediction'])
    #print(df_predictions)
    df_predictions.to_csv(out_predictions)    
    

#def test_performance(predictions):
    
    
    
if __name__ == '__main__':
    main()