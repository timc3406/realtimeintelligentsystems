from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
from sklearn import linear_model
from os import listdir
from os.path import isfile, join
from numpy import genfromtxt
import numpy as np

def update_model2(args):
    """
    update project model using the csv files, not jpg images
    """

    mypath = 'face_csv/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    X = None
    Y = []
    for file in onlyfiles:
        my_data = genfromtxt('face_csv/{}'.format(file), delimiter=',')
        if X is None:
            X = my_data
        else:
            X = np.concatenate((X, my_data))
        Yt = [file.split(".")[0] for _ in range(len(my_data)) ]
        Y.extend(Yt)

    recognizer = SVC(C=.5, kernel="linear", probability=True)
    recognizer.fit(X,Y)

    le = LabelEncoder()
    labels = le.fit_transform(Y)
    recognizer.fit(X,labels)
    print(recognizer.predict_proba(X[0].reshape(1, -1)))

    print("[INFO] training model...")

    f = open('output/recognizer.pickle', "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    f = open('output/le.pickle', "wb")
    f.write(pickle.dumps(le))
    f.close()
