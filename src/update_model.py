from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
from sklearn import linear_model


def update_model(args):
    """
    embed and create csv of the unknown images (randomized)
    """

    data = pickle.loads(open(args["embeddings"], "rb").read())

    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    print("[INFO] training model...")
    recognizer = SVC(C=0.1, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open(args["recognizer"], "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(args["le"], "wb")
    f.write(pickle.dumps(le))
    f.close()
