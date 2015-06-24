package org.apache.spark.mllib.clustering

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.scalatest.FunSuite
import org.apache.spark.mllib.util.TestingUtils._

class SpectralKMeansSuite extends FunSuite with MLlibTestSparkContext {
  import org.apache.spark.mllib.clustering.KMeans.{K_MEANS_PARALLEL, RANDOM}

  test("single cluster") {
    val data = sc.parallelize(Array(
      Vectors.dense(1.0, 2.0, 6.0, 1.0, 2.0, 6.0),
      Vectors.dense(1.0, 3.0, 0.0, 1.0, 3.0, 0.0),
      Vectors.dense(1.0, 4.0, 6.0, 1.0, 4.0, 6.0),
      Vectors.dense(1.3, 2.2, 6.4, 2.1, 2.3, 1.0),
      Vectors.dense(1.2, 3.0, 0.0, 1.0, 3.0, 0.0),
      Vectors.dense(1.1, 4.0, 6.2, 4.0, 4.0, 6.3),
      Vectors.dense(1.4, 2.5, 6.5, 1.2, 2.0, 6.2),
      Vectors.dense(1.5, 2.1, 0.6, 2.0, 3.0, 0.7),
      Vectors.dense(1.1, 4.6, 7.0, 1.5, 2.1, 6.0)
    ))

    //val center = Vectors.dense(1.0, 3.0, 4.0, 1.0, 3.0, 4.0)

    val model = SpectralKMeans.train(data, 1, 9, 0.2, 100, 1, K_MEANS_PARALLEL, 29)

    assert(model.predict(Vectors.dense(1.0, 2.0, 6.0, 1.0, 2.0, 6.0)) == 0)
    assert(model.predict(Vectors.dense(1.1, 4.6, 7.0, 1.5, 2.1, 6.0)) == 0)
    assert(model.predict(Vectors.dense(1.0, 3.0, 0.0, 1.0, 3.0, 0.0)) == 0)
    assert(model.predict(Vectors.dense(1.3, 2.2, 6.4, 2.1, 2.3, 1.0)) == 0)
/*
    println(model.clusterCenters.map(i => i.toArray.mkString(",")).mkString("\n"))

    val rdd_res = model.predict(data).collect()

    assert(rdd_res(0) == 0)
    assert(rdd_res(1) == 0)
    assert(rdd_res(2) == 0)
    assert(rdd_res(3) == 0)
    assert(rdd_res(4) == 0)
    assert(rdd_res(5) == 0)
    assert(rdd_res(6) == 0)
    assert(rdd_res(7) == 0)
    assert(rdd_res(8) == 0)*/
  }

  test("three clusters") {
    val data_raw = Array(
      Vectors.dense(-0.1, -2.0, -6.2, -1.2, -3.0, -5.9),
      Vectors.dense(0.1, 2.2, 6.1, 1.2, 3.2, 6.1),
      Vectors.dense(-0.05, -2.1, -6.4, -1.5, -2.5, -6.2),
      Vectors.dense(0.1, 1.9, 6.4, 1.1, 2.9, 6.1),
      Vectors.dense(10.2, 20.0, 16.0, 11.0, 23.0, 9.0),
      Vectors.dense(10.1, 24.0, 16.2, 11.5, 24.0, 9.3),
      Vectors.dense(9.4, 23.5, 16.5, 11.2, 22.0, 9.2),
      Vectors.dense(9.8, 21.1, 16.6, 12.0, 23.0, 9.7),
      Vectors.dense(11.1, 24.6, 17.0, 11.5, 22.1, 9.0))
    val data = sc.parallelize(data_raw)

    val model = SpectralKMeans.train(data, 3, 9, 0.4, 100, 1, K_MEANS_PARALLEL, 29)
    //val rdd_res = model.predict(data).collect()
    val result = for (i <- data_raw) yield{
      model.predict(i)
    }

  /* for (i <- result) {
      println(i)
    }
   System.exit(1)*/
    assert(result(0)==result(2))
    assert(result(1)==result(3))
    assert(result(4)==result(5))
    assert(result(5)==result(6))
    assert(result(6)==result(7))
    assert(result(7)==result(8))
    assert(result(0)!=result(1))
    assert(result(0)!=result(8))
    assert(result(1)!=result(8))

  }




}

