"""
This scales the landmarks from dlib, centering the face around the landmark 31,
and using the distance between x_40 and x_43 as a reference for the size of the face.

input file:
    FAME_landmarks.csv
output file:
    FAME_landmarks_scaled.csv
    
"""

import math
import pandas as pd

INPUT_FILE = 'landmarks_100.csv'
OUTPUT_FILE = 'landmarks_100_scaled.csv'

df = pd.read_csv(INPUT_FILE, skipinitialspace=True)
new_columns = df.columns.values
new_columns[0] = 'file'
df.columns = new_columns
df = df.set_index('file')

print(df.columns)

x_cols = [col for col in df.columns if 'X_' in col]
y_cols = [col for col in df.columns if 'Y_' in col]
all_cols = [col for col in df.columns if (('X_' in col or 'Y_' in col) and not 'eye' in col)]


for i in df.index:
    print(i)
    df.loc[i, x_cols] -= df.loc[i, 'X_30']
    df.loc[i, y_cols] -= df.loc[i, 'Y_30']
    
    difference = df.loc[i, 'X_42'] - df.loc[i, 'X_39']
    df.loc[i, all_cols] /= difference
    
print(df)
df.to_csv(OUTPUT_FILE)
    

#coords = []
#for i in df.iloc(0)[0]:
    #print(i)
    #n = 0
    #if i % 2 == 0:
        #n = i - 312 
    #if i % 2 != 0:
        #n = i - 176
        
    #n /= 68
    #coords.append(n)
    
#print(coords)
#df_new = pd.DataFrame(columns = df.columns)
#df_new.loc['test'] = coords
#df_new.to_csv("out.csv")