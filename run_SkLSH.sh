$SPARK_HOME/bin/spark-submit -v --executor-memory 160g --driver-memory 170g --conf "spark.driver.maxResultSize=15g" --class "com.Intel.bigDS.clustering.SkLSHTest" --master spark://sr471:7180 ./target/spectralkmeans.jar ${SPARK_ADDRESS} ${HDFS_ADDRESS}${DATA_FOLD}numerical.csv/ 224 0.0 1 320 0.002


