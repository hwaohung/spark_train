PATH=../spark-1.4.0-bin-hadoop2.4/bin:$PATH
PATH=../spark-1.4.0-bin-hadoop2.4/sbin:$PATH
host_name=spark://Kingdom:7077

# Stop the spark master
stop-master.sh

# Start the spark master
start-master.sh

# Detach 3 workers
spark-class org.apache.spark.deploy.worker.Worker $host_name -m 1G
#spark-class org.apache.spark.deploy.worker.Worker $host_name -m 1G &
#spark-class org.apache.spark.deploy.worker.Worker $host_name -m 1G &
