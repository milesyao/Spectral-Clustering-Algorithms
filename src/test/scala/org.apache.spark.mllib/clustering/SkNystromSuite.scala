package org.apache.spark.mllib.clustering

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.scalatest.FunSuite

class SkNystromSuite extends FunSuite with MLlibTestSparkContext {
  import org.apache.spark.mllib.clustering.KMeans.K_MEANS_PARALLEL
  /*
/*
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

    val model = SkNystrom.train(data, 1, 9, 0.5, 100, 1, K_MEANS_PARALLEL, 29)

    assert(model.predict(Vectors.dense(1.0, 2.0, 6.0, 1.0, 2.0, 6.0)) == 0)
    assert(model.predict(Vectors.dense(1.1, 4.6, 7.0, 1.5, 2.1, 6.0)) == 0)
    assert(model.predict(Vectors.dense(1.0, 3.0, 0.0, 1.0, 3.0, 0.0)) == 0)
    assert(model.predict(Vectors.dense(1.3, 2.2, 6.4, 2.1, 2.3, 1.0)) == 0)

    val rdd_res = model.predict(data).collect()

    assert(rdd_res(0) == 0)
    assert(rdd_res(1) == 0)
    assert(rdd_res(2) == 0)
    assert(rdd_res(3) == 0)
    assert(rdd_res(4) == 0)
    assert(rdd_res(5) == 0)
    assert(rdd_res(6) == 0)
    assert(rdd_res(7) == 0)
    assert(rdd_res(8) == 0)
  }*/

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

    val model = SkNystrom.train(data, 3, 9, 0.6, 100, 1, K_MEANS_PARALLEL, 29, 1.1)
    //val rdd_res = model.predict(data).collect()
   // println(model.pointProj.map(i => i._1 + "\n" + i._2).collect.mkString("\n"))
    //System.exit(1)

    val result = for (i <- data_raw) yield{
      model.predict(i)
    }

  for (i <- result) {
      println(i)
    }
   //System.exit(1)
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
*/
  test("pseudo data test(3 clusters") {
    val data_raw = Array(
      Vectors.dense(0.22927538045041368,0.15741606830001126,1.0369713094108526,0.5886610469170037),
      Vectors.dense(0.3697461267120109,0.23742639663845277,0.23721559687890895,0.9374812027962753),
      Vectors.dense(0.4523873822861232,0.33848311339230136,0.9325780980110324,0.56344690917107),
      Vectors.dense(0.3253983810338372,0.18084496120864135,0.23457784311942967,0.9079376841285469),
      Vectors.dense(0.4762152048670849,0.3393761589060125,1.0010751673490832,0.6829956713214204),
      Vectors.dense(0.32423838595647714,0.17526677898784962,0.9753434645980651,0.6570084462077806),
      Vectors.dense(0.41176592669032097,0.10208411586193863,0.9103247069654858,0.6532638214036393),
      Vectors.dense(0.3041785167695024,0.23210008647195668,0.20134230709923703,0.9777378733882757),
      Vectors.dense(0.34409839385592483,0.11311395349498107,0.9913566531874733,0.5699905192967161),
      Vectors.dense(0.3117268920773114,0.14185131241600005,0.3337013501174388,0.9913442419735675),
      Vectors.dense(0.33554994971096114,0.0426881343209959,1.010578292321262,0.635855385641787),
      Vectors.dense(0.25372325067404344,0.10135807330398297,1.0041768203887238,0.5643310301707342),
      Vectors.dense(0.42289518089230643,0.28173799513023373,0.8915905317871955,0.7284811180361569),
      Vectors.dense(0.2639435317781582,0.001755012700227973,0.9932990179374198,0.5733051594961326),
      Vectors.dense(0.28048084269231177,0.10522691355596442,0.9386244232405966,0.565799904541291),
      Vectors.dense(0.4132008566365232,0.13394444369600417,0.2687729966349543,0.9165352747323251)

    )
    val data = sc.parallelize(data_raw)

    val model = SkNystrom.train(data, 3, -1, 0.6, 100, 1, K_MEANS_PARALLEL, 29, 0.1)
    //val rdd_res = model.predict(data).collect()
    // println(model.pointProj.map(i => i._1 + "\n" + i._2).collect.mkString("\n"))
    //System.exit(1)

    val result = for (i <- data_raw) yield{
      model.predict(i)
    }

    for (i <- result) {
      println(i)
    }
    //System.exit(1)
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

