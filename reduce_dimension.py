import os
from sklearn.decomposition import PCA
from convert import *


def reduce_dimension(data, required):
    pca = PCA(n_components=required)
    X = pca.fit_transform([row[1] for row in data])

    for i in range(len(X)):
        data[i][1] = X[i]

    return data

def read_file(file_name):
    rows = list()
    fp = open(file_name, 'r')
    for line in fp:
        label, features = convert_features(line[:-2].split(','))
        rows.append([label, features])

    fp.close()
    return rows

def get_reduce_dimension_data(file_name, required):
    required = 10
    data = reduce_dimension(read_file(file_name), required)
    new_file = "{0}_reduce.txt".format(file_name)

    fp = open(new_file, 'w')
    for row in data:
        fp.write("{0} {1}".format(row[0], ' '.join([str(k) for k in row[1]])))
    fp.close()

    return new_file

if __name__ == "__main__":
    #get_reduce_dimension_data("kddcup.data.corrected", required=60)
    get_reduce_dimension_data("kddcup.data._10_percent", required=40)
