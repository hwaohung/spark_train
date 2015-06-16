from pyspark.mllib.regression import LabeledPoint

from attributes import *


def attr_vector(attr, attrs_map):
    vector = [0 for i in range(len(attrs_map))]
    vector[attrs_map[attr]] = 1
    return vector

def convert_features(row):
    vector = map(lambda x: float(x), row[0:1] + row[4:41])
    vector.extend(attr_vector(row[1], attr1s_map))
    vector.extend(attr_vector(row[2], attr2s_map))
    vector.extend(attr_vector(row[3], attr3s_map))
    return int(label_map[row[-1]]), vector

def write_to_file(rows, file_name):
    test = open(file_name, "w")
    for row in rows:
        label, features = convert_features(row)
        #print features
        test.write("{0} {1}\n".format(label, ' '.join([str(f) for f in features])))

    test.close()

def parsePoint(line):
    label = line[0]
    features = line[1]
    return LabeledPoint(label, features)

def parseRawPoint(line):
    line = line.split(' ')
    label = int(line[0])
    features = [float(line[i]) for i in range(1, len(line))]
    print label
    return LabeledPoint(label, features)

def get_data(sc, file_name):
    rdd = sc.textFile(file_name)
    rdd = rdd.map(lambda line: line[:-1].split(','))
    rdd = rdd.map(convert_features)
    return rdd.map(parsePoint)
 
#if __name__ == "__main__":
#    rdd = data_preprocessing("kddcup.data._10_percent")
#    for row in rdd.collect():
#        print row
#    #rows = read_file("kddcup.data._10_percent")
#    #rows = read_file("kddcup.data.corrected")
#    #write_to_file(rows, "Con.txt")
