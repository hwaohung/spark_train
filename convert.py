from attributes import *


def read_file(file_name):
    rows = list()

    fp = open(file_name, 'r')
    for line in fp:
    #for line in sc.textFile(file_name).collect():
        line = line[:-2]
        rows.append(line.split(','))

    fp.close()
    return rows

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
        test.write("{0} {1}\n".format(label, 
                                      ' '.join([str(f) for f in features])))
                                      #' '.join(["{0}:{1}".format(i, lp.features[i]) for i in range(len(lp.features)) if lp.features[i] != 0])))

    test.close()

if __name__ == "__main__":
    rows = read_file("kddcup.data._10_percent")
    #rows = read_file("kddcup.data.corrected")
    write_to_file(rows, "Con.txt")
