#from pyspark.mllib.classification import SVMModel
#from pyspark.mllib.classification import LogisticRegressionWithSGD
#from pyspark.mllib.classification import LogisticRegressionWithLBFGS
import numpy as np
import csv
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

def transform_label(labeled_point):
    if labeled_point.label == processed_label:
        labeled_point.label = 1.0
    else:
        labeled_point.label = 0.0

    return labeled_point


# Gen multiple classfiers
def gen_predictors(training_file):
    classifiers = dict()
    for item in label_map.iteritems():
        print "Gen predictor for label '{0}' ...".format(item[0])

        global processed_label
        processed_label = item[1]
        training_data = sc.textFile(training_file)
        training_data = training_data.map(parsePoint)
        training_data = training_data.map(transform_label)
        #training_data.foreach(transform_label)
        svm = SVMWithSGD.train(training_data)
        classifiers[item[1]] = svm
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
    N = len(label_map)
    # Generate the confusion matrix
    conf_mat = np.zeros((N, N))
    for pair in zip(actuals, predicts):
        conf_mat[pair[0], pair[1]] += 1
    
    # Transpose of confusion matrix 
    conf_mat_T = conf_mat.transpose()
    
    # Calculate the precision of each label
    precisions = list()
    for i in range(N):
        total = sum(conf_mat[i])
        if total != 0:
            precisions.append(conf_mat[i, i]/sum(conf_mat[i]))
        else:
            precisions.append(-1)
   
    # Calculate the recall of each label
    recalls = list()
    for i in range(N):
        total = sum(conf_mat_T[i])
        if total != 0:
            recalls.append(conf_mat_T[i, i]/sum(conf_mat_T[i]))
        else:
            recalls.append(-1)
            

    # Write out the confusion matrix to csv 
    titles = sorted(label_map.iteritems(), key=lambda x: x[1])
    writer = csv.writer(open("confusion_matrix.csv", "wb"), delimiter=',')
    writer.writerow([""] + [title for title in titles] + ["*precision"])

    for i in range(len(titles)):
        title = titles[i]
        if precisions[i] > 0: p_label = str(precisions[i])
        else: p_label = "N/A"
        writer.writerow([title] + [value for value in conf_mat[i]] + [p_label])

    temp = [precision for precision in precisions if precision > 0]
    precision = sum(temp) / len(temp)
    temp = [recall for recall in recalls if recall > 0]
    recall = sum(temp) / len(temp)

    row = ["*recall"]
    for recall in recalls:
        if recalls[i] > 0: r_label = str(recalls[i])
        else: r_label = "N/A"
        row.append(r_label)

    row.append("{0}/{1}".format(precision, recall))
    writer.writerow(row)
    

if __name__ == "__main__":
    training_file = "Con.txt"
    test_file = "Con.txt"
   
    #reduce_dimension(training_data, data, required=30)
 
    global classifiers, sorted_label
    classifiers = gen_predictors(training_file)
    sorted_label = get_sorted_label(training_file)

    actuals = sc.textFile(test_file).map(lambda x: int(x[:x.index(' ')]))
    predicts = sc.textFile(test_file).map(predict)

    report(actuals.collect(), predicts.collect())
