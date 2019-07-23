"""
Alternative approach to readjusting coordinates to 2D derotated landmarks
Scans frames for when the head pose is upright

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import shutil

IN_OPENFACE = '../landmarks/landmarks_100_scaled.csv'
OUT_FAILED = '../landmarks/failed_files_utk.csv'
OUT_LANDMARKS = '../landmarks/upright_landmarks_100_scaled.csv'

IN_FOLDER = '../image_samples/sample_1/sample_1/'
OUT_FOLDER = '../image_samples/sample_1_bad/'



df = pd.read_csv(IN_OPENFACE, skipinitialspace=True)
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

landmarks.to_csv(OUT_LANDMARKS)
#pd.DataFrame(failed_files).to_csv(OUT_FAILED)

for f in failed_files:
    print(f)
    try:
        shutil.move(IN_FOLDER+f+'.jpg', OUT_FOLDER+f+'.jpg')
    except FileNotFoundError:
        print("file does not exist")
