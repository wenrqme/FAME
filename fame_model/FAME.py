"""
This contains all the necessary functions to set up, train, run, and test the FAME model

if retraining everything, run logistic regression first (logreg_model/logreg.py)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import glob
import os

COEF = 'saved_model/linreg_coef_v3.txt'
INTERCEPT = 'saved_model/linreg_intercept_v3.txt'    

COEF_LOGREG = "logreg_model/saved_model/full_coef_v2.txt"
INTERCEPT_LOGREG = "logreg_model/saved_model/full_intercept_v2.txt"

DECEP_CL = "landmarks/cmd_low_upright_scaled.csv"
DECEP_CM = "landmarks/cmd_med_upright_scaled.csv"
DECEP_VM = "landmarks/vol_med_upright_scaled.csv"

def main():
    
    prepare_decep_data()
    
    # function calls to train FAME and test it on the 100 face test set
    landmarks = prepare_data("../mturk_surveys/survey2_mflikert/clean_results.csv", "landmarks/sample_2_lmk/", "landmarks/UTK_openface_landmarks_1000.csv")
    landmarks_upright = scan_upright_files(landmarks, "landmarks/UTK_openface_landmarks_1000_upright.csv")
    landmarks_upright_scaled = scale_landmarks(landmarks_upright, "landmarks/UTK_openface_landmarks_1000_scaled_upright.csv")
    coef, intercept = train(landmarks_upright_scaled)
    predictions = run_model("landmarks/upright_landmarks_100_scaled.csv", COEF, INTERCEPT, "predictions/linreg_predictions.csv")
    accuracy, mse = test_performance(predictions, 'landmarks/landmarks_100.csv', 'landmarks/landmarks_100_binarygenderlabels.csv')
    
    
    print("\n\n\n********************** SUMMARY **************************")
    print("100 face test set performance: ")
    print("accuracy: ", accuracy)
    print("mse: ", mse)
    
    run_decep_data(COEF, INTERCEPT)
    

#call this in the beginning so we only have to wait for scaling and scanning face landmarks once
def prepare_decep_data():
    landmarks_cls = pd.read_pickle("landmarks/cmd_low_frames_FAME.pkl.xz", compression='xz').rename(columns=lambda x: x.strip())
    landmarks_cms = pd.read_pickle("landmarks/cmd_med_frames_FAME.pkl.xz", compression='xz').rename(columns=lambda x: x.strip())
    landmarks_vms = pd.read_pickle("landmarks/vol_med_frames_FAME.pkl.xz", compression='xz').rename(columns=lambda x: x.strip())
    
    decep_landmarks_cls_upright = scan_upright_files(landmarks_cls, "landmarks/cmd_low_upright.csv")    
    decep_landmarks_cms_upright = scan_upright_files(landmarks_cms, "landmarks/cmd_med_upright.csv")
    decep_landmarks_vms_upright = scan_upright_files(landmarks_vms, "landmarks/vol_med_upright.csv")    
    decep_landmarks_cls_scaled = scale_landmarks(decep_landmarks_cls_upright, "landmarks/cmd_low_upright_scaled.csv")
    decep_landmarks_cms_scaled = scale_landmarks(decep_landmarks_cms_upright, "landmarks/cmd_med_upright_scaled.csv")
    decep_landmarks_vms_scaled = scale_landmarks(decep_landmarks_vms_upright, "landmarks/vol_med_upright_scaled.csv")
    

#run the FAME and logistic regression models on the deception data and test the performcance
def run_decep_data(coef, intercept):
    predictions_cls = run_model(DECEP_CL, coef, intercept, "predictions/decep_predictions_cmd_low.csv")
    predictions_cms = run_model(DECEP_CM, coef, intercept, "predictions/decep_predictions_cmd_med.csv")
    predictions_vms = run_model(DECEP_VM, coef, intercept, "predictions/decep_predictions_vol_med.csv")
    logreg_predictions_cls = run_logreg_model(DECEP_CL, COEF_LOGREG, INTERCEPT_LOGREG, "predictions/logreg_predictions_cmd_low.csv")
    logreg_predictions_cms = run_logreg_model(DECEP_CM, COEF_LOGREG, INTERCEPT_LOGREG, "predictions/logreg_predictions_cmd_med.csv")
    logreg_predictions_vms = run_logreg_model(DECEP_VM, COEF_LOGREG, INTERCEPT_LOGREG, "predictions/logreg_predictions_vol_med.csv")
    
    labels_cls = format_labels(predictions_cls, logreg_predictions_cls, "labels/labels_cmd_low.csv")
    labels_cms = format_labels(predictions_cms, logreg_predictions_cms, "labels/labels_cmd_med.csv")
    labels_vms = format_labels(predictions_vms, logreg_predictions_vms, "labels/labels_vol_med.csv")
   
    # test performance on these labels (may do differently from before because there's no logreg labels currently)
    #TODO: add step to combine files (use combine_files())
    labels_combined = combine_files(labels_cls, labels_cms, labels_vms)
    df_labels =  decep_combine_gender_FAME(labels_combined) #temporarily just look at this one file
    FAME_acc, FAME_acc_f, FAME_acc_m = decep_performance(df_labels, 'FAME')
    logreg_acc, logreg_acc_f, logreg_acc_m = decep_performance(df_labels, 'logreg')   
    
    print("\n\n\n********************** DECEPTION SUMMARY **************************")
    print("\ndeception data test set performance (FAME): ")
    print("accuracy: ", FAME_acc)
    print("accuracy females: ", FAME_acc_f)
    print("accuracy males: ", FAME_acc_m)
    
    print("\ndeception data test set performance (logreg): ")
    print("accuracy: ", logreg_acc)
    print("accuracy females: ", logreg_acc_f)
    print("accuracy males: ", logreg_acc_m)
    

def run_test_data(coef, intercept):
    predictions = run_model("landmarks/upright_landmarks_100_scaled.csv", coef, intercept, "predictions/linreg_predictions_training.csv")
    accuracy, mse = test_performance(predictions, 'landmarks/landmarks_100.csv', 'landmarks/landmarks_100_binarygenderlabels.csv')
    
    
    print("\n\n\n********************** TEST SUMMARY **************************")
    print("100 face test set performance: ")
    print("accuracy: ", accuracy)
    print("mse: ", mse)    


# combines 3 csv files into one
def combine_files(file_1, file_2, file_3):
    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)
    df3 = pd.read_csv(file_3)
    
    df_combined = df1.append([df2, df3])
    combined_file = "labels/decep_labels.csv"
    df_combined.to_csv(combined_file)
    return combined_file


# linear regression training  L1 regularization and 10-fold cross validation
# the model weights and intercept are saved to text files
# takes in a scaled landmarks file
def train(landmarks_file):
    #OUT_COEF = 'saved_model/linreg_coef_v3.txt'
    #OUT_INTERCEPT = 'saved_model/linreg_intercept_v3.txt'
    OUT_MSE = 'saved_model/mse_folds_v3.txt'
    niter = 10
    
    # we will use sklearn LASSO model which has a hyhperparameter: alpha 
    # alpha: Constant that multiplies the L1 term. Defaults to 1.0. 
    # alpha = 0 is equivalent to an ordinary least squares
    alphas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 3,10,100,1000]
    
    df = pd.read_csv(landmarks_file, skipinitialspace=True) # replace with file with human labeled data
    print('data loaded')
    
    all_features = list(df.columns.values)[2:]
    landmark_cols = [feat for feat in all_features if (('X_' in feat or 'Y_' in feat) and not 'eye' in feat)]
    features = landmark_cols
    target = "rating"
    
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
    y = df[target]
    
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
            
            
            #test on test set and deception data
            #save temporary coefficients and intercept
            temp_coef = "saved_model/temp_coef.txt"
            temp_intercept = "saved_model/temp_intercept.txt"
            np.savetxt(temp_coef, np.array([lasso2.coef_]))
            np.savetxt(temp_intercept, np.array([best_model.intercept_]))         
            run_test_data(temp_coef, temp_intercept)
            run_decep_data(temp_coef, temp_intercept)
    
    
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
    
    np.savetxt(COEF, best_model.coef_)
    np.savetxt(INTERCEPT, np.array([best_model.intercept_]))
    np.savetxt(OUT_MSE, ave_test_mse)
    return COEF, INTERCEPT


# runs the FAME model (from the coef and intercept files) on landmarks_file
# and saves the predictions
def run_model(landmarks_file, coef_file, intercept_file, out_predictions):
    #out_predictions = "predictions/linreg_predictions.csv"
    #load df,  coef_, and intercept_
    df = pd.read_csv(landmarks_file, index_col=[0])
    coefficients = np.loadtxt(coef_file)
    intercepts = np.loadtxt(intercept_file)
    
    urls = np.array(df.index)
    #print('data loaded')
    
    model = LinearRegression()
    coefficients = [coefficients]
    model.coef_ = np.array(coefficients)
    model.intercept_ = np.array(intercepts)
    model.classes_ = np.array([0,1]) #0 for male, 1 for female
    
    #X = df.values
    all_features = list(df.columns.values)[2:]
    landmark_cols = [feat for feat in all_features if (('X_' in feat or 'Y_' in feat) and not 'eye' in feat)]
    features = landmark_cols 
    X = df[features]
    #print(X)
    #print(X.shape)
    
    prediction = model.predict(X)
    #print("prediction: ", prediction)
    
    prediction_format = np.zeros(len(prediction))
    for i in range(len(prediction)):
        prediction_format[i] = prediction[i][0]
    #print(prediction_format)
    
    df_predictions = pd.DataFrame({'image_url': urls, 'prediction': prediction_format}, columns=['image_url', 'prediction'])
    #print(df_predictions)
    df_predictions.to_csv(out_predictions)
    return out_predictions


#run logreg model
def run_logreg_model(landmarks_file, coef_file, intercept_file, out_predictions):
    #load df,  coef_, and intercept_
    df = pd.read_csv(landmarks_file, index_col=[0]) #use text.csv for 100 face file, test1000.csv for 1000 face file
    coefficients = np.loadtxt(coef_file)
    intercepts = np.loadtxt(intercept_file)
    urls = np.array(df.index)
    #print('data loaded')
    
    model = LogisticRegression()
    coefficients = [coefficients]
    model.coef_ = np.array(coefficients)
    model.intercept_ = np.array(intercepts)
    model.classes_ = np.array([0,1]) #0 for male, 1 for female
    
    all_features = list(df.columns.values)[2:]
    landmark_cols = [feat for feat in all_features if (('X_' in feat or 'Y_' in feat) and not 'eye' in feat)]
    features = landmark_cols 
    X = df[features]
    #print('\n')
    #print(X)
    #print(X.shape)
    
    prediction = model.predict_proba(X)
    #print("\nprediction: ", prediction)
    
    prediction_format = np.zeros(len(prediction))
    for i in range(len(prediction)):
        prediction_format[i] = prediction[i][1] # predicstion gives [probability male, probability female]
    #print(prediction_format)
    
    df_predictions = pd.DataFrame({'image_url': urls, 'prediction': prediction_format}, columns=['image_url', 'prediction'])
    #print(df_predictions)
    df_predictions.to_csv(out_predictions)
    return out_predictions
    
    
# combines the openface csv files and adds a column in front for for the FAME rating
def prepare_data(FAME_ratings_file, openface_folder, output_landmark_file):
    df_ratings = pd.read_csv(FAME_ratings_file, skipinitialspace=True)
    df_ratings.set_index(['Image_filename'], inplace=True)
    
    list_ = []
    i = 1;
    
    for file in glob.glob(openface_folder + "*.csv"):
        df = pd.read_csv(file, skipinitialspace=True, index_col=None, header=0)
        filename = os.path.splitext(file)[0][23:]
        print(filename)
        try:
            df.insert(0, 'gender', filename[3])
            df.insert(0, 'rating', df_ratings.loc[filename, 'Answer.Rating'])
            df.insert(0, 'Filename', filename)
            list_.append(df)
        except KeyError:
            print("not in mturk file")
        #print(i)
        i += 1
        
    landmarks = pd.concat(list_, axis = 0, ignore_index = True)
    #print(len(landmarks))
    
    landmarks.to_csv(output_landmark_file, encoding='utf-8',index=False)
    return landmarks
    

# scales the landmarks from openface, centering the face around the nose,
# and using the distance between the eyes as a reference for the size of the face.
def scale_landmarks(landmarks_df, scaled_landmark_file):
    df = landmarks_df

    new_columns = df.columns.values
    new_columns[0] = 'Filename'
    df.columns = new_columns
    df = df.set_index('Filename')
    
    #print(df.columns)
    
    x_cols = [col for col in df.columns if 'X_' in col]
    y_cols = [col for col in df.columns if 'Y_' in col]
    all_cols = [col for col in df.columns if (('X_' in col or 'Y_' in col) and not 'eye' in col)]
    #print(all_cols)
    
    for i in df.index:
        #print(i)
        df.loc[i, x_cols] -= df.loc[i, 'X_30']
        df.loc[i, y_cols] -= df.loc[i, 'Y_30']
        
        difference = df.loc[i, 'X_42'] - df.loc[i, 'X_39']
        df.loc[i, all_cols] /= difference
        
    #print(df)
    df.to_csv(scaled_landmark_file)
    return scaled_landmark_file
    
    
# formats labels for the deception data from the FAME predictions
def format_labels(linreg_pred, logreg_pred, out_labels):
    df1 = pd.read_csv(linreg_pred)
    df2 = pd.read_csv(logreg_pred)
    
    df1.set_index(df1.columns[0], inplace=True)
    #df2.set_index(df1.columns[0], inplace=True)
    
    #organize by face, so that individual frames are not considered
    df1_new = pd.DataFrame(columns = ['image_filename'])
    df1_new['image_filename'] = df1['image_url'].str[:-13]
    df1_new = pd.concat([df1_new, df1], axis = 1)
    del df1_new['image_url']
    #print(df1_new)
    
    df1_new = (df1_new.set_index(['image_filename',df1_new.groupby('image_filename').cumcount()])['prediction']
            .unstack()
            .add_prefix('FAME_')
            .reset_index())
    
    #add average column
    df1_ave = df1_new.copy()
    df1_ave['FAME'] = df1_ave.mean(numeric_only=True, axis=1)
    df1_ave['FAME_variance'] = df1_ave.var(numeric_only=True, axis=1)
    #print(df1_new)
    
    #logreg model
    df2_new = pd.DataFrame(columns = ['image_filename'])
    df2_new['image_filename'] = df2['image_url'].str[:-13]
    df2_new = pd.concat([df2_new, df2], axis = 1)
    del df2_new['image_url']
    
    df2_new = (df2_new.set_index(['image_filename',df2_new.groupby('image_filename').cumcount()])['prediction']
            .unstack()
            .add_prefix('logreg_')
            .reset_index())
    
    df2_ave = df2_new.copy()
    df2_ave['logreg'] = df2_ave.mean(numeric_only=True, axis=1)
    df2_ave['logreg_variance'] = df2_ave.var(numeric_only=True, axis=1)
    
    #concatenate them
    df3 = pd.concat([df1_ave, df2_ave], axis = 1)
    df3 = df3.loc[:,~df3.columns.duplicated()]
    #print(df3)
    
    df3.to_csv(out_labels)
    return out_labels
    

#scan for upright face files
def scan_upright_files(in_openface_landmarks_df, out_landmarks):
    df = in_openface_landmarks_df
    #print(df.head())
    df.rename(columns={'file':'Filename'})
    #print(df.columns)
    grouped = df.groupby(['Filename'])
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
                #print('index: ', index)
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
    pd.DataFrame(failed_files).to_csv(out_landmarks + '-FAILED.csv')
    return landmarks


# look at the performance of linear regression on FAME ratings
def test_performance(predictions_file, ratings_file, genders_file):
    #load data from files
    df_predictions = pd.read_csv(predictions_file)
    df_lm_fast = pd.read_csv(ratings_file)
    df_lm_gender = pd.read_csv(genders_file)
    df_ratings = df_lm_fast['rating']
    df_true_gender = df_lm_gender['gender']
    
    df = pd.concat([df_predictions, df_ratings, df_true_gender], axis = 1)
    
    #is there a non manual way to concatrnate df of different sizes?
    df = df [:60]
    
    #Look at accuracy at predicting gender, using <3.5 as male and >3.5 as female
    _sum = 0
    for i in range(len(df.index)):
        if (df.loc[i, 'prediction'] < 3.5 and df.loc[i, 'gender'] == 0) or (df.loc[i, 'prediction'] >= 3.5 and df.loc[i, 'gender'] == 1):
            _sum += 1
    accuracy = _sum/len(df.index)
    
    #print(accuracy)   
    
    #Look at MSE separately for males and females
    
    df_m= df.loc[df['gender'] == 0] #males
    mse_m = mean_squared_error(df_m['rating'], df_m['prediction'])
    #print("\nmale mean squared error: ", mse_m)
    
    df_f= df.loc[df['gender'] == 1] #females
    mse_f = mean_squared_error(df_f['rating'], df_f['prediction'])
    #print("female mean squared error: ", mse_f)
    
    mse_all = mean_squared_error(df['rating'], df['prediction'])
    print("both mean squared error: ", mse_all)
    
    #Look at variance of scores
    
    all_var = df['prediction'].var()
    f_var = df_f['prediction'].var()
    m_var = df_m['prediction'].var()
    
    #print('\nvariance: ', all_var, 
    #      '\nvariance males: ', m_var, 
    #      '\nvariance females: ', f_var)
    
    return accuracy, mse_all


# make a dataframe that has all the information about gender and FAME scores of the data
# fro use later in performance testing
def decep_combine_gender_FAME(labels_file):
    GENDER_FILE_1 = 'labels/gender_data.csv'
    GENDER_FILE_2 = 'labels/old_data.csv'
    GENDER_FILE_3 = 'labels/hs_avg.csv'
    
    # load data
    
    all_labels = pd.read_csv(labels_file)
    df_labels = all_labels[['image_filename', 'FAME', 'logreg']] #only look at final FAME and logreg scores
    df_labels.to_csv("labels/decep_labels_FAME.csv")
    df_1 = pd.read_csv(GENDER_FILE_1)
    df_2 = pd.read_csv(GENDER_FILE_2)
    df_3 = pd.read_csv(GENDER_FILE_3, skipinitialspace=True)
    #print('data loaded')
    
    #Future warning, this line is to temporarily fix the SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.
    df_labels.is_copy = False
    
    #df_labels['filename'] = df_labels['image_filename'].str.split('\\').str[1]
    df_labels['root'] = df_labels['image_filename'].str.rsplit('-', 3).str[0]
    df_labels['user_id'] = df_labels['image_filename'].str.rsplit('-', 1).str[1]
    #print(df_labels)
    df_i = df_labels[df_labels['image_filename'].str.contains('I')] # interrogator
    df_w = df_labels[df_labels['image_filename'].str.contains('W')] # witness
    
    df_gender = pd.concat([df_1, df_2])
    df_gender.set_index(['root'], inplace=True)    
    
    df_3['is_male'].apply(int)
    
    
    file_genders =  pd.Series(df_3['is_male'].values, index=df_3['Filename'].str.rsplit('-', 1).str[1].str.split('.').str[0]).to_dict()
    #display(file_genders)
    
    # low stakes labels
    df_4 = pd.read_csv('labels/commanded_low_stakes_genders.csv')
    gender_dict =  pd.Series(df_4['is_male'].values, index=df_4['filename'].str.rsplit('-', 1).str[1].str.split('.').str[0]).to_dict()
    file_genders.update(gender_dict)
    
    df_labels['is_male'] = np.nan
    df_labels['is_male'] = df_labels['user_id'].astype(str).map(file_genders)    
    
    #adding gender labels in using df_gender
    #display(df_gender)
    count = 0
    
    for i in df_labels.index:
        if df_labels['image_filename'].str.contains('I')[i]:
            try:
                df_labels.loc[i, 'is_male'] = df_gender.loc[df_labels.loc[i,'root'], 'interrogator_is_male']
                count += 1
            except KeyError:
                #print('not in table: ', df_labels.loc[i,'root'])
                continue
            #print('i')
        elif df_labels['image_filename'].str.contains('W')[i]:
            try:
                df_labels.loc[i, 'is_male'] = df_gender.loc[df_labels.loc[i,'root'], 'witness_is_male']
                count += 1
            except KeyError:
                #print('not in table: ', df_labels.loc[i,'root'])
                continue
            #print('w')
    
    #print(count)
    df_labels.to_csv("labels/decep_gender_labels.csv")
    return df_labels
    

# look at how well fame did in scoring the deception data
# model can be either 'FAME' or 'logreg'
def decep_performance(df_labels, model):
    # Look at variance off scores for males and females
    
    # females
    df_f= df_labels.loc[df_labels['is_male'] == 0]
    var_f = df_f[model].var()
    mean_f = df_f[model].mean()
    #print('variance females: ', var_f, '\nmean females: ', mean_f)
    
    # males
    df_m= df_labels.loc[df_labels['is_male'] == 1]
    var_m = df_m[model].var()
    mean_m = df_m[model].mean()
    #print('variance males: ', var_m, '\nmean males: ', mean_m)
    
    overall_mean = df_labels[model].mean()
    print('overall mean: ', overall_mean)    
    
    #Look at accuracy at predicting gender, using <mean as male and >mean as female
    #for logreg, 0.5 is the cutoff
    cutoff = 0.5
    if model == "FAME":
        cutoff = overall_mean
    print("cutoff: ", cutoff)
    
    _sum = 0
    _sum_f = 0
    _sum_m = 0
    for i in df_f.index:
        if (df_f.loc[i, model] >= cutoff and df_f.loc[i, 'is_male'] == 0):
            _sum += 1
            _sum_f += 1
    for i in df_m.index:
        if (df_m.loc[i, model] < cutoff and df_m.loc[i, 'is_male'] == 1):
            _sum += 1
            _sum_m += 1
    accuracy = _sum/(len(df_f.index)+len(df_m.index))
    
    accuracy_f = _sum_f/len(df_f.index)
    accuracy_m = _sum_m/len(df_m.index)
    
    #print(accuracy)
    #print("f: ", accuracy_f)
    #print("m: ", accuracy_m)
    
    return accuracy, accuracy_f, accuracy_m


if __name__ == '__main__':
    main()
    