"""This program combines the openface csv files and adds a column in front for
for the FAME rating

"""

import pandas as pd
import glob
import os


FAME_file = "../../mturk_surveys/survey1_mflikert/clean_results.csv"
df_ratings = pd.read_csv(FAME_file)
df_ratings.set_index(['Image_filename'], inplace=True)

list_ = []
i = 1;

for file in glob.glob("sample_1_lmk/*.csv"):
    df = pd.read_csv(file, index_col=None, header=0)
    filename = os.path.splitext(file)[0][13:]
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

landmarks.to_csv('landmarks_100_binarygenderlabels.csv', encoding='utf-8',index=False)
