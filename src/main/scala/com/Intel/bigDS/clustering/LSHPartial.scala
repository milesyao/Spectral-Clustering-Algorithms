package com.Intel.bigDS.clustering


import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.KMeans.{K_MEANS_PARALLEL, RANDOM}
import java.io.PrintWriter

import scala.util.Random

//This program is extracted from org.apache.spark.mllib.SkLSH in convenience for local debug
//Critical random projection progress
object LSHPartial extends Serializable {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("LSH Partial Test")
    @transient val sc = new SparkContext(conf)

    val out = new PrintWriter("./LSHPartial.log")

    val data_raw = Array(
      Vectors.dense(-0.1, -2.0, -6.2, -1.2, -3.0, -5.9),
      Vectors.dense(0.1, 2.2, 6.1, 1.2, 3.2, 6.1),
      Vectors.dense(-0.05, -2.1, -6.4, -1.5, -2.5, -6.2),
      Vectors.dense(0.1, 1.9, 6.4, 1.1, 2.9, 6.1),
      Vectors.dense(10.2, 20.0, 16.0, 11.0, 23.0, 9.0),
      Vectors.dense(10.1, 24.0, 16.2, 11.5, 24.0, 9.3),
      Vectors.dense(9.4, 23.5, 16.5, 11.2, 22.0, 9.2),
      Vectors.dense(9.8, 21.1, 16.6, 12.0, 23.0, 9.7),
      Vectors.dense(11.1, 24.6, 17.0, 11.5, 22.1, 9.0)
    )
    val data = sc.parallelize(data_raw)

    var numDims = -1

    if (numDims < 0) numDims = data.count().toInt
    //val parseddata = data.map(_.toArray).zipWithIndex().map(_.swap)
    val parseddata = data.zipWithIndex()
    val colnum = data.first.size
    val summary = Statistics.colStats(data)
    val colmax = summary.max.toArray
    val colmin = summary.min.toArray

    out.println("col max")
    out.println(colmax.mkString(","))
    out.println("col min")
    out.println(colmin.mkString(","))

    val M = (math.log(numDims)).ceil.toInt
    val colspan = colmax.zip(colmin).map(i => math.abs(i._1-i._2))
    val spansum = colspan.sum
    val colpossibility = colspan.map(i => i / spansum)
    val colinterval = colspan.map(i => i / 20)

    out.println("col interval")
    out.println(colinterval.mkString(","))

    var incrementalvalue = 0.0
    val colincremental = for (i <- colpossibility) yield {
      incrementalvalue = incrementalvalue + i
      incrementalvalue
    }

    out.println("incremental value")
    out.println(colincremental.mkString(","))

    val rand_seed = new Random(3)
    val hyperplanes = for (i <- (0 until M).toArray) yield {
      val rand_num = rand_seed.nextDouble
      var res = 0
      val incrementallen = colincremental.length
      var break_flag = false
      for (j <- 0 until incrementallen if break_flag!=true) {
        if (j == 0) {
          if (rand_num >= 0 && rand_num < colincremental(j)) {res = 0; break_flag = true;}
        }
        else {
          if (rand_num >= colincremental(j-1) && rand_num <= colincremental(j)) {res = j; break_flag = true;}
        }
      }
     res
    }

    val colmin_br = sc.broadcast(colmin)
    val colinterval_br = sc.broadcast(colinterval)

    val datawithbank = data.map(i => i.toArray.zipWithIndex).map(i => i.map(j => {
      var res = 0
      var low = 0.0
      var high = 0.0
      val colmin_node = colmin_br.value
      val colinterval_node = colinterval_br.value
      var break_flag = false
      for (m <- 0 until 20 if break_flag!=true) {
        val lower = colmin_node(j._2) + colinterval_node(j._2) * m
        var upper = 0.0
        if (m==19) upper = lower + colinterval_node(j._2) + 0.1
        else upper = lower + colinterval_node(j._2)
        if (j._1>=lower && j._1<=upper) {
          res = m
          break_flag = true
        }
      }
      res
    }))
    out.println("data with bank")
    out.println(datawithbank.map(i => i.mkString(",")).collect.mkString("\n"))

    val databanksummary = datawithbank.treeAggregate(Array.ofDim[Int](colnum, 20))(
      seqOp =  (U, V) => {
        for (i <- 0 until V.size) {
          U(i)(V(i)) = U(i)(V(i)) + 1
        }
        U
      },
      combOp = (U1, U2) => {
        U1.zip(U2).map(i => i._1.zip(i._2).map(j => j._1 + j._2))
      }
    )
    out.println("data bank summary")
    out.println(databanksummary.map(i => i.mkString(",")).mkString("\n"))

    val minbin = databanksummary.map(i => i.indexOf(i.min))

    out.println("data minimal bin")
    out.println(minbin.mkString(","))

    val thresholds = colmin.zip(colinterval).zip(minbin).map{case(((min,interval), bin)) => min + bin * interval}
    val thresholds_view = for (i <- hyperplanes) yield {
      thresholds(i)
    }


    out.println("hyperplanes")
    out.println(hyperplanes.mkString(","))
    out.println("thresholds")
    out.println(thresholds_view.mkString(","))
    out.println("M = " + M)



    val lsh = new com.Intel.bigDS.hash.LSH(parseddata, hyperplanes, thresholds, M)
    val model = lsh.run()
    val simiData = model.clusters.flatMap{i => {
      val res = for (m <- i._2) yield {
        val data = for (n <- i._2.toArray) yield {
          Vectors.sqdist(m._1, n._1)
        }
        (m._2, Vectors.sparse(numDims, i._2.map(_._2.toInt).toArray, data))
      }
      res
    }
    }

    out.println(simiData.map(i => i._1 + " : " + i._2.toArray.mkString(",")).collect.mkString("\n"))
    out.close()
  }
}