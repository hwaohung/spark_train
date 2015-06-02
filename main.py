#from pyspark.mllib.classification import SVMModel
#from pyspark.mllib.classification import LogisticRegressionWithSGD
#from pyspark.mllib.classification import LogisticRegressionWithLBFGS
#from pyspark.mllib.util import MLUtils
#import org.apache.log4j.Logger
#import pysparkorg.apache.log4j.Level

import csv
import numpy as np
from collections import Counter


from pyspark.mllib.classification import SVMWithSGD
from pyspark import SparkContext
from pyspark import SparkConf

from convert import *
from reduce_dimension import *

#Logger.getLogger("org").setLevel(Level.ERROR)
#Logger.getLogger("akka").setLevel(Level.ERROR)

app_name = "WordCount"
spark_master = "spark://Kingdom:7077"
spark_home = "../spark-1.3.1-bin-hadoop2.4"

conf = SparkConf()
conf.setMaster(spark_master)
conf.setSparkHome(spark_home)
conf.setAppName(app_name)
conf.set("spark.executor.memory", "1g")
#conf.set("spark.akka.frameSize", "100")
#conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
#conf.set("spark.kryoserializer.buffer.mb", "64")
#conf.set("spark.executor.extraJavaOptions", "-XX:+UseCompressedOops")
#conf.set("spark.storage.memoryFraction", "0.6")
sc = SparkContext(conf=conf, pyFiles=["main.py", "convert.py"])


# Return the sorted label, and the weight list
def get_sorted_label(training_data):
    rdd = training_data.map(lambda x: int(x.label))
    items = sorted(rdd.countByValue().items(), key=lambda x: x[0])

    return [item[0] for item in items], [item[1] for item in items]

def transform_label(labeled_point):
    if labeled_point.label == processed_label:
        labeled_point.label = 1.0
    else:
        labeled_point.label = 0.0

    return labeled_point

# Gen multiple classfiers
def gen_predictors(training_data):
    classifiers = dict()
    for item in label_map.iteritems():
        print "Gen predictor for label '{0}' ...".format(item[0])

        global processed_label
        processed_label = item[1]
        svm = SVMWithSGD.train(training_data.map(transform_label))
        classifiers[item[1]] = svm

    return classifiers

def predict(features):
    candidates = list()
    weights = list()
    for i in range(len(sorted_label)):
        label = sorted_label[i]
        if classifiers[label].predict(features) == 1:
            candidates.append(label)
            weights.append(label_weights[i])
 
    total = sum(weights)
    # When not fit label, then directly asign to min count label 
    if len(candidates) == 0: belong = sorted_label[0]
    else: belong = choice(candidates, p=[float(weight)/total for weight in weights])
    return belong

def report(labelsAndPreds):
    N = len(label_map)
    # Generate the confusion matrix
    conf_mat = np.zeros((N, N))
    for pair in labelsAndPreds:
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
   
    #print precisions
    #print recalls 
    # Write out the confusion matrix to csv 
    titles = [item[0] for item in sorted(label_map.iteritems(), key=lambda x: x[1])]
    writer = csv.writer(open("confusion_matrix.csv", "wb"), delimiter=',')
    writer.writerow([""] + [title for title in titles] + ["*precision"])

    for i in range(len(titles)):
        title = titles[i]
        if precisions[i] >= 0: p_label = str(precisions[i])
        else: p_label = "N/A"
        writer.writerow([title] + [value for value in conf_mat[i]] + [p_label])

    temp = [precision for precision in precisions if precision >= 0]
    precision = sum(temp) / len(temp)
    temp = [recall for recall in recalls if recall >= 0]
    print temp
    recall = sum(temp) / len(temp)

    row = ["*recall"]
    for value in recalls:
        if value >= 0: r_label = str(value)
        else: r_label = "N/A"
        row.append(r_label)

    row.append("{0}/{1}".format(precision, recall))
    writer.writerow(row)
    

if __name__ == "__main__":
    original_file = "kddcup.data.corrected"

    #data = get_data(sc, original_file)
    data = get_reduce_dimension_data(sc, original_file, required=60)
    exit()
    splits = data.randomSplit([0.6, 0.4], 10)
    training_data = splits[0].cache()
    test_data = splits[1]
 
    global classifiers, sorted_label, label_weights
    classifiers = gen_predictors(training_data)
    training_data
    sorted_label, label_weights = get_sorted_label(training_data)

    labelsAndPreds = test_data.map(lambda x: (int(x.label), predict(x.features)))

    report(labelsAndPreds.collect())
