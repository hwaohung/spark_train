#from pyspark.mllib.classification import SVMModel
#from pyspark.mllib.classification import LogisticRegressionWithSGD
#from pyspark.mllib.classification import LogisticRegressionWithLBFGS
import numpy as np
from sklearn.decomposition import PCA

from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark.mllib.util import MLUtils
import copy


app_name = "WordCount"
spark_master = "spark://zhangqingjundeMacBook-Air.local:7077"
spark_home = "/Users/johnny/Desktop/spark-1.1.1-bin-hadoop2.4"

sc = SparkContext(spark_master, app_name, spark_home, batchSize=-1)

attr1s = ["udp", "icmp", "tcp"]
attr1s_map = { attr1s[i]: i for i in range(len(attr1s)) }

attr2s = ["domain", "netbios_ssn", "urp_i", "Z39_50", "smtp", "gopher",
          "private", "echo", "printer", "red_i", "eco_i", "sunrpc",
          "ftp_data", "urh_i", "pm_dump", "pop_3", "pop_2", "systat",
          "ftp", "uucp", "whois", "netbios_dgm", "efs", "remote_job",
          "sql_net", "daytime", "ntp_u", "finger", "ldap", "netbios_ns",
          "kshell", "iso_tsap", "ecr_i", "nntp", "shell", "domain_u",
          "uucp_path", "courier", "exec", "tim_i", "netstat", "telnet",
          "rje", "hostnames", "link", "auth", "http_443", "csnet_ns",
          "X11", "IRC", "tftp_u", "imap4", "supdup", "name",
          "nnsp", "mtp", "http", "bgp", "ctf", "klogin",
          "vmnet", "time", "discard", "login", "other", "ssh"
         ]
attr2s_map = { attr2s[i]: i for i in range(len(attr2s)) }

attr3s = ["OTH", "RSTR", "S3", "S2", "S1", "S0", "RSTOS0", "REJ", "SH", "RSTO", "SF"]
attr3s_map = { attr3s[i]: i for i in range(len(attr3s)) }

labels = [ "back", "buffer_overflow", "ftp_write", "guess_passwd", "imap", "ipsweep",
           "land", "loadmodule", "multihop", "neptune", "nmap", "normal",
           "perl", "phf", "pod", "portsweep", "rootkit", "satan",
           "smurf", "spy", "teardrop", "warezclient", "warezmaster"
         ]
label_map = { labels[i]: i for i in range(len(labels)) }

def transform_to_data(rows):
    test = open("Con.txt", "w")
    data = list()
    for row in rows:
        lp = LabeledPoint(*convert_features(row))
        data.append(lp)
        test.write("{0} {1}\n".format(lp.label, 
                                      ' '.join([str(f) for f in lp.features])))
                                      #' '.join(["{0}:{1}".format(i, lp.features[i]) for i in range(len(lp.features)) if lp.features[i] != 0])))

    test.close()
    return data

def convert_features(row):
    vector = map(lambda x: float(x), row[0:1] + row[4:41])
    vector.extend(attr_vector(row[1], attr1s_map))
    vector.extend(attr_vector(row[2], attr2s_map))
    vector.extend(attr_vector(row[3], attr3s_map))
    return label_map[row[-1]], vector

def attr_vector(attr, attrs_map):
    vector = [0 for i in range(len(attrs_map))]
    vector[attrs_map[attr]] = 1
    return vector

def read_file(file_name):
    rows = list()
    fp = open(file_name, 'r')
    
    for line in fp:
        # Remove ".\n"
        line = line[0:-2]
        rows.append(line.split(','))

    fp.close()
    return rows

def get_principal_indexes(data, required):
    pca = PCA(n_components=required)
    X = np.array([row.features for row in data])
    pca.fit(X)
    return pca

# Gen multiple classfiers
def gen_predictors(training_data):
    classifiers = dict()
    for label in label_map.values():
        training_data = copy.deepcopy(training_data)
        for row in training_data:
            if row.label == label: row.label = 1
            else: row.label = 0

        svm = SVMWithSGD.train(sc.parallelize(training_data))
        classifiers[label] = svm

    return classifiers

def predict(test_data, classifiers, hist):
    # Min count label(in training data)
    min_label = hist.index(min(hist))
    predicts = list()
    for row in test_data:
        belong = None
        for label in label_map.values():
            if classifiers[label].predict(row.features) == 1:
                if belong is None: belong = label
                elif hist[label] > hist[belong]: belong = label
            
        if belong is None: belong = min_label
        predicts.append(belong)

    return predicts

def reduce_dimension(training_data, data, required=30)
    # Dimension reduce
    pca = get_principal_indexes(training_data, required=required)
    pca.fit_transform(data)

def report(actuals, predicts):
    tp_count = dict()
    actual_count = dict()
    predict_count = dict()

    # Initialize the count
    for label in label_map.values():
        tp_count[label] = 0
        actual_count[label] = 0
        predict_count[label] = 0

    for i in range(len(actuals)):
        actual = actuals[i]
        predict = predicts[i]

        actual_count[actual] += 1
        predict_count[predict] += 1
        if predict == actual:
            tp_count[actual] += 1
    
    p_amount = 0.0
    precision = 0.0
    r_amount = 0.0
    recall = 0.0
    for label in label_map.values():
        if actual_count[label] != 0: 
            precision += tp_count[label] / float(actual_count[label])
            p_amount += 1

        if predict_count[label] != 0: 
            recall += tp_count[label] / float(predict_count[label])
            r_amount += 1

    print p_amount, r_amount
    precision /= float(p_amount)
    recall /= float(r_amount)
        
    print "Precision: {0}".format(precision)
    print "Recall: {0}".format(recall)


if __name__ == "__main__":
    rows = read_file("10_percent.txt")
    data = transform_to_data(rows)
    #data = MLUtils.loadLibSVMFile(sc, "Con.txt").collect()
    #data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt").collect()

    

    training_data = data[:10000]
    test_data = data[:10000]
    
    #reduce_dimension(training_data, data, required=30)

    hist = [0 for label in range(len(label_map))]
    for label in sorted(label_map.values()):
        hist[label] += 1
        
    classifiers = gen_predictors(training_data)

    actuals = [row.label for row in test_data]
    predicts = predict(test_data, classifiers, hist)

    report(actuals, predicts)
