$SPARK_HOME/bin/spark-submit -v --executor-memory 160g --driver-memory 170g --class "com.Intel.bigDS.clustering.SpectralKMeansTest" --master ${SPARK_ADDRESS} ./target/spectralkmeans.jar ${SPARK_ADDRESS} ${HDFS_ADDRESS}${DATA_FOLD}numerical.csv/ 256 0.002 1 320

