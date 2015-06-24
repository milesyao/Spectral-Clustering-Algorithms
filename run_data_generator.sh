$SPARK_HOME/bin/spark-submit -v --executor-memory 160g --driver-memory 170g --class "com.Intel.bigDS.clustering.DataGenerator" --master ${SPARK_ADDRESS} ./target/spectralkmeans.jar ${SPARK_ADDRESS} ${HDFS_ADDRESS}${DATA_FOLD} 256 256000 64 numerical.csv 320 0.0002

