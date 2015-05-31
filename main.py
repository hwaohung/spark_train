from pyspark.mllib.classification import SVMModel
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext


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
    #test = open("Con.txt", "w")
    data = list()
    for row in rows:
        lp = LabeledPoint(*convert_features(row))
        data.append(lp)
        #test.write(",".join([str(x) for x in convert_features(row)])+"\n")

    #test.close()
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

if __name__ == "__main__":
    rows = read_file("10_percent.txt")
    data = transform_to_data(rows)
    #help(SVMModel)
    svm = SVMWithSGD.train(sc.parallelize(data))
    print svm.predict(data[0].features)
