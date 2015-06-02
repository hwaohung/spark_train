import os
from sklearn.decomposition import PCA
from convert import *


def reduce_dimension(data, required):
    pca = PCA(n_components=required)
    X = pca.fit_transform([row[1] for row in data])
    Y = [row[0] for point in data]
    return X, Y

def read_file(file_name):
    rows = list()
    fp = open(file_name, 'r')
    for line in fp:
        label, features = convert_features(line[:-2].split(','))
        rows.append((label, features))

    fp.close()
    return rows

def get_reduce_dimension_data(file_name, required):
    data = read_file(file_name)
    new_file = "{0}_reduce.txt".format(file_name)
    X, Y = reduce_dimension(data, required)
    del data

    fp = open(new_file, 'w')
    for i in range(len(X)):
        fp.write("{0} {1}".format(Y[i], ' '.join(X[i])))
    fp.close()

    return new_file

if __name__ == "__main__":
    get_reduce_dimension_data("kddcup.data.corrected", required=60)
