from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import glob
import os
from model.config import *
import numpy as np


def train_svm():
    pos_feat_path = '../data/features/pos'
    neg_feat_path = '../data/features/neg'

    # Classifiers supported
    clf_type = 'LIN_SVM'

    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
    print(np.array(fds).shape, len(labels))
    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        print("Training a Linear SVM Classifier")
        clf.fit(fds, labels)
        joblib.dump(clf, model_path)
        print("Classifier saved to {}".format(model_path))


train_svm()
