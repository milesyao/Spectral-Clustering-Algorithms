
package com.Intel.bigDS.clustering


import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.mllib.clustering.KMeans.{K_MEANS_PARALLEL, RANDOM}


/**
 * Created by yaochunnan on 5/29/15.
 * Test of Spectral Clustering based on LSH(Random projection) on Spark.
 *
 * Parameters: address of spark master, address of data on HDFS, number of partitions, sigma is scalaing parameter converting distance matrix to similarity matrix, "sub partial" indicates t-nearrest ratio
 */

object SkLSHTest extends Serializable {
  def run(args: Array[String]): (Array[(Int, Vector)],Array[(Int, Int)]) = {
    println("Spectral KMeans method on Synthetic data")
    if (args.length != 6) {
      System.err.println("ERROR:Spectral Clustering: <spark master> <path to data> <nParts> <sigma> <number of clusters> <sub partial>")
    }
    println("===========================" + args.mkString(",") + "===============================")
    val conf = new SparkConf()
      .setMaster(args(0))
      .setAppName("Spectral Clustering with LSH(RP)")
    @transient val sc = new SparkContext(conf)

    val data_address = args(1)
    val nParts = args(2).toInt
    val sigma = args(3).toDouble
    val numcluster = args(4).toInt
    val subpartial = args(5).toDouble
    val parsed = sc.textFile(data_address, nParts).map(_.split(",").map(_.toDouble)).map(Vectors.dense(_)).distinct.cache()

    val numDim = parsed.count
    val numfeatures = parsed.first.size

    val start = System.currentTimeMillis / 1000

    val model = SkLSH.train(parsed, numcluster, numDim.toInt, sigma, subpartial, 100, 1, K_MEANS_PARALLEL, 29)
    val predictions = model.predictall()

    val end = System.currentTimeMillis / 1000

    val meta_res = predictions.map(i => (i._2,i._1) ).groupByKey


    val C = meta_res.count().toInt
    val N = parsed.count

    //compute centers in original data
    val centers = meta_res.map(i => {
      val NN = i._2.size
      (i._1, i._2.foldLeft[Array[Double]](new Array[Double](numfeatures))((U, V) => V.toArray.zip(U).map(i => i._1+i._2)).map(i => i/NN))
    }).collect.map(i => (i._1, Vectors.dense(i._2))).toMap

    val center_num = meta_res.map(i => (i._1, i._2.size)).collect()


    val centers_br = sc.broadcast(centers)
    val meta_dis = meta_res.map{ i =>
      val centers_node = centers_br.value(i._1)
      (i._1, i._2.map(j => Vectors.sqdist(j, centers_node)))
    }

    val ASE_res = meta_dis.map(i => math.pow(i._2.sum,2)).sum / N
    val WSSSE = meta_dis.map(i => i._2.sum).sum

    val cluster_avg = meta_dis.map(i => (i._1, i._2.sum / i._2.size)).collect

    val meta_DBI = for (i <- 0 until C) yield {
      var max_meta = 0.0
      for (j <- 0 until C if j!=i) {
        val m = cluster_avg(i)
        val n = cluster_avg(j)
        val meta_res = (m._2 + n._2) / Vectors.sqdist(centers(m._1), centers(n._1))
        if (meta_res > max_meta) max_meta = meta_res
      }
      max_meta
    }
    val DBI_res = meta_DBI.sum / C

    println("*********************************************************************************")
    println("*********************************************************************************")
    println("Training costs " + (start - end) + " seconds")
    println("*********************************************************************************")
    println("*********************************************************************************")
    println(ASE_res)
    println("ASE value")
    println(DBI_res)
    println("DBI value")
    println("WSSSE="+WSSSE)

    sc.stop()

    (centers.toArray,center_num)
    /*
    val labelrdd = parseddata.map(_.label)

    val valuesAndPreds = model.predictall(featurerdd)
         .join(featurerdd.zip(labelrdd)).map(i => i._2).groupByKey.map(i => (i._1, i._2.map((_,1))
         .groupBy(_._1).map(j => (j._1, j._2.map(_._2).sum)).toMap)).collect
*/
   /* val valuesAndPreds = parseddata.map { point =>
      val prediction = model.predict(point.features)
      (prediction, point.label)
    }.groupByKey.map(i => (i._1, i._2.map((_,1)).groupBy(_._1).map(j => (j._1, j._2.map(_._2).sum)).toMap)).collect
   */
    //val ACC = valuesAndPreds.collect.sum.toDouble / numDim
   // println("==============Clustering Quality Observation===============")
    //println(valuesAndPreds.mkString("\n"))
    //println("Accuracy of clustering is " + ACC + ".")
    //model.predict(parseddata.map(_.features)).map(_.toDouble)

  }
  def main(args: Array[String]): Unit = {
    run(args)
  }
}
