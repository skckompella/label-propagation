import scipy.io
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

def label_propagate(affinity_mat, row_sum_diag_mat, label_ids, labels):
    print " > Initializing......",
    Y = np.zeros(labels.shape)
    for i in l:
        Y[i] = labels[i]
    print "[done]"


    rand_walk_mat = np.linalg.inv(row_sum_diag_mat).dot(affinity_mat)
    #print label_ids
    for i in range(0,1000):
        #print " Itr: " + str(i) + "\r"
        Y = rand_walk_mat.dot(Y)
        for i in labeled_ids:
            Y[i] = labels[i]  # Clamping
        #row_sum = Y_mod.sum(axis=0)
    #if abs(row_sum) < 0.003:
     #   return Y
    return Y

def knn():
    return


## Main##

print " > Loading data......",
data = scipy.io.loadmat('SSL,set=9,data.mat')
feature_mat = data['X'].toarray()
labels = data["y"]
print "[done]"

print " > Loading data2.....",

data2 = scipy.io.loadmat('SSL,set=9,splits,labeled=10.mat')
labeled_ids = data2['idxLabs']
unlabeled_ids = data2['idxUnls']
print "[done]"

print " > Computing W.......",
affinity_mat = kneighbors_graph(feature_mat, 5, mode='connectivity', include_self=True).toarray()
affinity_mat = affinity_mat + affinity_mat.transpose() ## To make the matrix symmetric
affinity_mat[np.nonzero(affinity_mat)] = 1 ## any non zero value has to be 1
print "[done]"

print " > Computing D.......",
row_sums = affinity_mat.sum(axis=1)
row_sum_diag_mat = np.diag(row_sums)
print "[done]"

accuracy = []

for l in labeled_ids:
    print " > Beginning Label prop..."
    Y_mod = label_propagate(affinity_mat, row_sum_diag_mat, l, labels)
    y = Y_mod
    a = 0
    negc = 0
    posc = 0
    for i in range(Y_mod.shape[0]):
        if Y_mod[i] <= 0:
            y[i] = -1
            negc += 1
            if labels[i] ==  -1:
                a += 1
        else:
            y[i] = 1
            posc += 1
            if labels[i] ==  1:
                a += 1
    print "Neg: " + str(negc)
    print "Pos: " + str(posc)
    print "Accuracy = " + str(float(a)/1500)
    accuracy.append(a)


#x = [i for i in range(0,1500)]

#fig2 = plt.figure()
#bx = plt.gca()
#bx.scatter(x, Y_mod)
#plt.show()