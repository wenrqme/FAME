import numpy as np
import os
import shutil

#os.mkdir('samples')

# list all files in dir
files = [f for f in os.listdir('.') if os.path.isfile(f)]

# select 0.1 of the files randomly 
random_files = np.random.choice(files, 1000, replace = False)

print(random_files)

for i in range(1000):
    print("copied ", i)
    shutil.copy(random_files[i], 'sample2')