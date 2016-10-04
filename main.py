import scipy.io
import numpy as np
from sklearn import neighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import sys

def check_accuracy(predicted_labels, given_labels, initially_labeled=0):
    right_predictions = np.sum(predicted_labels == given_labels)
    return float(right_predictions+initially_labeled)/given_labels.shape[0]


def label_propagate(rand_walk_mat, labels, label_ids):
    Y = np.zeros(labels.shape)
    for i in label_ids:
        Y[i] = labels[i]

    for i in range(0,10):
        Y = rand_walk_mat.dot(Y)
        for i in label_ids:
            Y[i] = labels[i]  # Clamping
    Y[Y<0] = -1
    Y[Y>=0] = 1
    return Y

def knn(feature_mat, labels, ids):
    ft = np.ndarray(shape=(ids.shape[0],feature_mat.shape[1]))
    l = np.ndarray(shape=(ids.shape[0],1))
    for i in range(0, ids.shape[0]):
        ft[i] = feature_mat[ids[i]]
        l[i]     = labels[ids[i]]


    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(ft, l.ravel())
    predictions = knn.predict(feature_mat[10:])
    predictions = predictions.reshape((labels[10:].shape[0], 1)) #change from (1490, ) to (1490, 1)
    return predictions


def label_moons():
    moons = scipy.io.loadmat('2moons.mat')
    moon_ft = moons['x']
    moon_labels = moons['y']
    moon_labels_select = moon_labels[:10]
    affinity_mat = kneighbors_graph(moon_ft, 5, mode='connectivity', include_self=True).toarray()
    affinity_mat = affinity_mat + affinity_mat.transpose()  ## To make the matrix symmetric
    affinity_mat[np.nonzero(affinity_mat)] = 1  ## any non zero value has to be 1
    row_sums = affinity_mat.sum(axis=1)
    row_sum_diag_mat = np.diag(row_sums)
    rand_walk_mat = np.linalg.inv(row_sum_diag_mat).dot(affinity_mat)
    Y = label_propagate(rand_walk_mat, moon_labels, range(0,10))
    y = moons['x'][:, 1].transpose()
    x = moons['x'][:, 0].transpose()
    plt.scatter(x, y, c=Y)
    #plt.show()
    print "Two moons accuracy: ",
    print str(check_accuracy(Y, moon_labels )*100) + "%"
    #return moons_ft, moon_labels_select


def label_benchmark():
    data = scipy.io.loadmat('SSL,set=9,data.mat')
    feature_mat = data['X'].toarray()
    labels = data["y"]

    data2 = scipy.io.loadmat('SSL,set=9,splits,labeled=10.mat')
    labeled_ids = data2['idxLabs']
    unlabeled_ids = data2['idxUnls']
    sigma = 2
    #affinity_mat = kneighbors_graph(feature_mat, 5, mode='distance', include_self=False).toarray()
    affinity_mat = kneighbors_graph(feature_mat, 5, mode='connectivity', include_self=True).toarray()
    #affinity_mat = np.divide(affinity_mat,(sigma**2))
    affinity_mat = affinity_mat + affinity_mat.transpose()  ## To make the matrix symmetric
    affinity_mat[np.nonzero(affinity_mat)] = 1  ## any non zero value has to be 1

    row_sums = affinity_mat.sum(axis=1)
    row_sum_diag_mat = np.diag(row_sums)

    rand_walk_mat = np.linalg.pinv(row_sum_diag_mat).dot(affinity_mat)

    lp_accuracy = []
    knn_accuracy = []

    for split in labeled_ids:  # Each split is a sequence of label IDs
        lp_predictions = label_propagate(rand_walk_mat, labels, split)
        lp_accuracy.append(check_accuracy(lp_predictions, labels))
        knn_predictions = knn(feature_mat, labels, split)
        knn_accuracy.append(check_accuracy(knn_predictions, labels[10:], 10))

    print "Label proagation accuracy on benchmark.pdf : ",
    print str(np.mean(lp_accuracy)*100) + "%"
    print "KNN Accuracy on benchmark.pdf: ",
    print str(np.mean(knn_accuracy)*100) + "%"



## Main##
label_moons()
label_benchmark()

