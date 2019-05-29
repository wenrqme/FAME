# FAME (Facial Androgyny Measure & Estimation)
FAME is a tool that estimates the masculinity/femininity of a human face using facial landmark information from dlib. FAME uses a scale of 0 (most masculine) to 6 (most feminine). This can be helpful in examining how gender appearance on a scale affects behavior, rather than just considering binary gender.

FAME was trained using linear regression (sklearn.linear_model.Lasso) with 10-fold cross validation. FAME used a sample of ~1000 images from the UTKFace dataset for training. The FAME scores used for training were obtained from mturk worker ratings of the images.

## Contents

#### fame_model:
**FAME_training.py**: used for training the FAME model. Input is a landmark file. Outputs a coefficient, intercept, and scale factors file, located in the folder **saved_model**.
**FAME_run.py**: runs a trained model on a landmark mile. Input is landmark, coefficient, interectpt, and scale factors files. Outputs the predictions in the folder **predictions**.
**FAME_performance_analysis.ipynb**: analyzes prediction results from the model, and gives information on error, gender classifcication accuracy, and results of the most masculine & feminine faces
**important_landmarks_v4**: a visual of the 8 landmarks that had weights of the highest magnitude
**landmarks**: contains files for landmark detection using dlib. **face_landmark_detection_FAME.py** builds off of existing code from http://dlib.net/face_landmark_detection.py.html and outputs a csv file with the filenames, FAME ratings, and landmark information. **face_landmark_detection.py** does the same thing but without the FAME ratings

#### mturk_surveys:
contains results and analyses of 2 mturk batch surveys for FAME ratings.

## Notes
Some variables use "FAST" rather than "FAME". The project was initially called FAST (Facial Androgyny Scale Test), and was changed, but those variable names were not yet updated.
