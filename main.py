#from pyspark.mllib.classification import SVMModel
#from pyspark.mllib.classification import LogisticRegressionWithSGD
#from pyspark.mllib.classification import LogisticRegressionWithLBFGS
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA

from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.util import MLUtils
from attributes import *
import copy


app_name = "WordCount"
spark_master = "spark://zhangqingjundeMacBook-Air.local:7077"
spark_home = "/Users/johnny/Desktop/spark-1.1.1-bin-hadoop2.4"

conf = SparkConf()
conf.setMaster(spark_master)
conf.setSparkHome(spark_home)
conf.setAppName(app_name)
conf.set("spark.executor.memory", "4g")
conf.set("spark.akka.frameSize", "100")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.kryoserializer.buffer.mb", "64")
conf.set("spark.executor.extraJavaOptions", "-XX:+UseCompressedOops")
# Lower best
conf.set("spark.storage.memoryFraction", "0.6")
sc = SparkContext(conf=conf)


def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

def get_principal_indexes(data, required):
    pca = PCA(n_components=required)
    X = [[row.features for row in data]]
    pca.fit(X)
    return pca

# Sort the label by the count in increasing order
def get_sorted_label(file_name):
    rp = open(file_name, 'r')
    labels = list()
    for line in rp:
        idx = line.index(' ')
        labels.append(int(line[:idx]))

    rp.close()
    hist = Counter(labels)
    sorted_items = sorted(hist.iteritems(), key=lambda x: x[1])
    return [item[0] for item in sorted_items]

# Gen multiple classfiers
def gen_predictors(training_file):
    classifiers = dict()
    for key in label_map.keys():
        print "Gen predictor for label '{0}' ...".format(key)

        training_data = sc.textFile(file_name).map(parsePoint)
        training_data = sc.textFile(file_name).map(lambda x: 1 if x.label == label else 0)
        svm = SVMWithSGD.train(training_data)
        classifiers[label_map[key]] = svm
        del training_data

    return classifiers

def predict(line):
    belong = None
    features = [float(x) for x in line.split(' ')][1:]
    for i in range(len(sorted_label)-1, -1, -1):
        label = sorted_label[i]
        if classifiers[label].predict(features) == 1:
            belong = label
            break
       
    # When not fit label, then directly asign to min count label 
    if belong is None: belong = sorted_label[0]
    return belong

def reduce_dimension(training_data, data, required=30):
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

    conf_mat = np.zeros((len(items), len(items)))
    for i in range(len(actuals)):
        actual = actuals[i]
        predict = predicts[i]

        conf_mat[actual, predict] += 1
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
   
    items = sorted(label_map.iteritems(), key=lambda x: x[1])
    writer = csv.writer(open("confusion_matrix.csv", "wb"), delimiter=',')
    writer.writerow([""] + ["{0}({1})".format(*item) for item in items])
    for item in items:
        writer.writerow(["{0}({1})".format(*item)] + [value for value in conf_mat[item[1]]])
    
    print "Precision: {0}".format(precision)
    print "Recall: {0}".format(recall)


if __name__ == "__main__":
    training_file = "Con.txt"
    test_file = "Con.txt"
   
    #reduce_dimension(training_data, data, required=30)
 
    global classifiers, sorted_label
    classifiers = gen_predictors(training_file)
    sorted_label = get_sorted_label(training_file)

    actuals = sc.textFile(test_file).map(lambda x: int(x[:x.index(' ')]))
    predicts = sc.textFile(test_file).map(predict)

    actuals = actuals.collect()
    predicts = predicts.collect()

    #print "Actuals"
    #print Counter(actuals)
    #print "Predicts"
    #print Counter(predicts)

    report(actuals, predicts)
