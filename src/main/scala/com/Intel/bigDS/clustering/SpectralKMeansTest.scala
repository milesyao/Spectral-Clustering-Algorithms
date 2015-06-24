package com.Intel.bigDS.clustering


import org.apache.spark.mllib.clustering.SpectralKMeans
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.mllib.clustering.KMeans.{K_MEANS_PARALLEL, RANDOM}
import java.io.PrintWriter

/**
 * Test of Spectral Clustering based on t-nearrest neighbors.
 *
 * Parameters: address of Spark master, path to data on HDFS, number of partitions, sparsity is the ratio of t-nearest neighbors, sigma is scalaing parameter used to convert distance matrix to similarity matrix, numger of clusters
 */
object SpectralKMeansTest extends Serializable {
  val NametoLabel = Map("C15" -> 0, "CCAT" -> 1, "E21" -> 2, "ECAT" -> 3, "GCAT" -> 4, "M11" -> 5)
  def run(args: Array[String]): (Array[(Int, Vector)],Array[(Int, Int)]) = {
    println("Spectral KMeans method on Synthetic data")
    if (args.length != 6) {
      System.err.println("ERROR:Spectral Clustering: <spark master> <path to data> <nParts> <sparsity> <sigma> <number of clusters")
    }
    println("===========================" + args.mkString(",") + "===============================")
    val conf = new SparkConf()
      .setMaster(args(0))
      .setAppName("Spectral Clustering")
    @transient val sc = new SparkContext(conf)

    val data_address = args(1)
    val nParts = args(2).toInt
    val sparsity = args(3).toDouble
    val sigma = args(4).toDouble
    val numcluster = args(5).toInt
    val br_nametolabel = sc.broadcast(NametoLabel)
    val parsed = sc.textFile(data_address, nParts).map(_.split(",").map(_.toDouble)).map(Vectors.dense(_)).distinct.repartition(256)cache()
    val numDim = parsed.count
    val numfeatures = parsed.first.size
    val start = System.currentTimeMillis / 1000
    val model = SpectralKMeans.train(parsed, numcluster, numDim.toInt, sparsity, 100, 1, K_MEANS_PARALLEL, 29)
    val predictions = model.predictall()
    val end = System.currentTimeMillis / 1000
    val meta_res = predictions.map(i => (i._2,i._1) ).groupByKey

    val C = meta_res.count().toInt
    val N = parsed.count

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

    (centers.toArray,center_num)


  }
  def main(args: Array[String]): Unit = {
    run(args)
  }
}

