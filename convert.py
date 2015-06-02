from numpy.random import choice

try:
    from pyspark.mllib.regression import LabeledPoint
except Exception as ex:
    print ex

attr1s = ["udp", "icmp", "tcp"]
attr1s_map = { attr1s[i]: i for i in range(len(attr1s)) }

attr2s = [
            "urp_i", "netbios_ssn", "Z39_50", "tim_i", "smtp",
            "domain", "private", "echo", "printer", "red_i",
            "eco_i", "sunrpc", "ftp_data", "urh_i", "pm_dump",
            "pop_3", "pop_2", "systat", "ftp", "uucp",
            "whois", "tftp_u", "netbios_dgm", "efs", "remote_job",
            "sql_net", "daytime", "ntp_u", "finger", "ldap",
            "netbios_ns", "kshell", "iso_tsap", "ecr_i", "nntp",
            "http_2784", "shell", "domain_u", "uucp_path", "courier",
            "exec", "aol", "netstat", "telnet", "gopher",
            "rje", "hostnames", "link", "ssh", "http_443",
            "csnet_ns", "X11", "IRC", "harvest", "imap4",
            "supdup", "name", "nnsp", "mtp", "http",
            "bgp", "ctf", "klogin", "vmnet", "time",
            "discard", "login", "auth", "other", "http_8001"
         ]
attr2s_map = { attr2s[i]: i for i in range(len(attr2s)) }

attr3s = [
            "OTH", "RSTR", "S3", "S2", "S1",
            "S0", "RSTOS0", "REJ", "SH", "RSTO",
            "SF"
         ]
attr3s_map = { attr3s[i]: i for i in range(len(attr3s)) }

labels = [ "back", "buffer_overflow", "ftp_write", "guess_passwd", "imap", "ipsweep",
           "land", "loadmodule", "multihop", "neptune", "nmap", "normal",
           "perl", "phf", "pod", "portsweep", "rootkit", "satan",
           "smurf", "spy", "teardrop", "warezclient", "warezmaster"
         ]
label_map = { labels[i]: i for i in range(len(labels)) }


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
    return LabeledPoint(int(line[0]), [float(line[i]) for i in range(1, len(line))])

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
