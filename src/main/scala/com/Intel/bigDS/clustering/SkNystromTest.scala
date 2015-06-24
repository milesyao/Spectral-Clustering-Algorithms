package com.Intel.bigDS.clustering


import org.apache.spark.mllib.clustering.{SkNystrom, SpectralKMeans}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.KMeans.{K_MEANS_PARALLEL, RANDOM}
import java.io.PrintWriter

/**
 * Spectral Clustering based on Nystrom Optimization.
 * This implementation is too unstable to use.
 *
 */

object SkNystromTest extends Serializable {
  val NametoLabel = Map("C15" -> 0, "CCAT" -> 1, "E21" -> 2, "ECAT" -> 3, "GCAT" -> 4, "M11" -> 5)
  def main(args: Array[String]): Unit = {
    println("Spectral KMeans using Nystrom method on synthetic data")
    if (args.length != 6) {
      System.err.println("ERROR:Spectral Clustering: <spark master> <path to data> <nParts> <sparsity> <sigma> <num of cluster>")
    }
    println("===========================" + args.mkString(",") + "===============================")
    val conf = new SparkConf()
      .setMaster(args(0))
      .setAppName("Spectral Clustering with Nystrom optimization")
    @transient val sc = new SparkContext(conf)


    val out = new PrintWriter("/home/yilan/GenBase/SpectralKMeans/watchout_SkNystrom.log")
    val data_address = args(1)
    val nParts = args(2).toInt
    val sparsity = args(3).toDouble
    val sigma = args(4).toDouble
    val numcluster = args(5).toInt
    val br_nametolabel = sc.broadcast(NametoLabel)
    val parsed = sc.textFile(data_address, nParts).map(_.split(",").map(_.toDouble)).map(Vectors.dense(_)).distinct.cache()
    out.println("all data (only for small tests)")
    out.println(parsed.map(i => i.toArray.mkString(",")).collect.mkString("\n"))
/*
    //parse data
    val parsed = sc.textFile(data_address, nParts)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .mapPartitions { iter =>
      val ntl = br_nametolabel.value
      iter.map { line =>
        val items = line.split(' ')
        val label = ntl(items.head).toDouble
        val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
          val indexAndValue = item.split(':')
          val index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
        val value = indexAndValue(1).toDouble
          (index, value)
        }.unzip
        (label, indices.toArray, values.toArray)
      }
    }
    val numFeatures = {
      parsed.persist(StorageLevel.MEMORY_ONLY)
      parsed.map { case (label, indices, values) =>
        indices.lastOption.getOrElse(0)
      }.reduce(math.max) + 1
    }
    val parseddata = parsed.map { case (label, indices, values) =>
      LabeledPoint(label, Vectors.sparse(numFeatures, indices, values))
    }*/
    //val featurerdd = parseddata.map(_.features).distinct
    val numDim = parsed.count
    val numfeatures = parsed.first.size
    val model = SkNystrom.train(parsed, numcluster, numDim.toInt, sparsity, 100, 1, K_MEANS_PARALLEL, 29, sigma)
    out.println(model.clusterCenters.map(i => i.toArray.mkString(",")).mkString("\n"))
    out.println("cluster centers")
    out.println(model.clusterCenters.length)
    out.println("cluster centers number")
    val predictions = model.predictall()
    val meta_res = predictions.map(i => (i._2,i._1) ).groupByKey
    out.println(meta_res.collect.map(i => i._1 + " : " + i._2.size).mkString("\n"))
    out.println("grouped data")

    val C = meta_res.count().toInt
    val N = parsed.count

    val centers = meta_res.map(i => {
      val NN = i._2.size
      (i._1, i._2.foldLeft[Array[Double]](new Array[Double](numfeatures))((U, V) => V.toArray.zip(U).map(i => i._1+i._2)).map(i => i/NN))
    }).collect.map(i => (i._1, Vectors.dense(i._2))).toMap
   // println(centers.map(i => i._1 + " : " + i._2.toArray.mkString(",")).mkString("\n"))
   // println(meta_res.count)
   // println("cneter num")
    //System.exit(1)
    val centers_br = sc.broadcast(centers)
    val meta_dis = meta_res.map{ i =>
      val centers_node = centers_br.value(i._1)
      (i._1, i._2.map(j => Vectors.sqdist(j, centers_node)))
    }

    val ASE_res = meta_dis.map(i => math.pow(i._2.sum,2)).sum / N
    val WSSSE = meta_dis.map(i => i._2.sum).sum

    //val cluster_avg = meta_dis.map(i => (i._1, i._2.sum / i._2.size)).collect.sortBy(_._1).map(_._2)
    val cluster_avg = meta_dis.map(i => (i._1, i._2.sum / i._2.size)).collect
    out.println(cluster_avg.map(i => i._1 + "," + i._2 ).mkString("\n"))
    out.println("C=" + C)
   // System.exit(1)
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

    out.println(ASE_res)
    out.println("ASE value")
    out.println(DBI_res)
    out.println("DBI value")
    out.close()
    println("WSSSE="+WSSSE)





    //println(model.predictall(parsed).map(i => (i._2, 1)).groupBy(_._1).map(i => (i._1, i._2.map(_._2).sum)).collect.toMap.mkString(","))
    //println(model.pointProj.map(i => i._1.toArray.mkString(",") + "   |   " + i._2.toArray.mkString(",")).collect.mkString("\n"))
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
}