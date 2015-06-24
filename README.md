Spectral Clustering Algorithm examples
======================================

This package contains source codes of our initial implementation of several versions of spectral clustering algorithms on Apache Spark.They are:

	1. Parallel Spectral Clustering based on t-nearrest neighbors(PSC)
	2. Parallel Spectral Clustering based on Nystrom optimization(NYSC)
	3. Parallel Spectral Clustering based on Locality Sensitive Hashing(DASC)

Run
===

You need to install Spark 1.3.0 or higher versions together with hadoop 1.0.4 as storage support. 

After building with sbt, use spark-submit tool to submit applications to Spark cluster. 

	Step1: Generate data using "DataGenerator".
	Step2: Run KMeansTest, SkLSHTest, SkNystromTestor SpectralKMeansTest and see their processing time and clustering accuracy(WSSE value). KMeansTest directly calls Spark MLlib's KMeans class. It is used as a reference for clustering quality measurement.  
        Step3: Change algorithm parameters to test their performance under different circumstances. 

Example shell scripts are located under the root directory(run_data_generator.sh, run_KMeansTest.sh, run_SkLSH.sh, run_SpectralKMeans.sh). Please change the server address to your real condition. 

For details on meanings of parameters and theoretical backgrounds of each implementation, please refer to the comments in souce codes. 

Good Luck.

Correctness
===========

Tests are passed for correctness of these implementations. However, the Nystrom optimization method regularly throw errors when data is big. PSC and LSH method works well. LSH shows the best performance among all the three implementations. 

Contact
=======

Please contact yaochunnan@gmail.com for  bugs or questions. 


