package com.Intel.bigDS.clustering

import breeze.numerics.abs
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import scala.collection.mutable.ArrayBuffer
import scala.sys.process._
import scala.util.Random
import java.io.PrintWriter


/**
 * Created by yaochunnan on 5/19/15.
 * In this program, we generate pseudo clustering data using random numbers. First we generate
 * clustering centers randomly. Then based on these centers and indicated variance given by user,
 * we randomly generate data points surrounding its corresponding center.
 *
 * This function receives eight parameters:1.address of spark master 2.path of data storage at HDFS 3. number of points 4. length of features 5.table name at HDFS 6.number of clusters 7.value of variance
 */

object DataGenerator {

  def getRangeRDD(sc: SparkContext, rand: Random, size: Int, parts: Int): RDD[(Int, Int, Int)] = {

    val avg = size / parts
    val fullNum = size % parts
    val seedArray = ArrayBuffer[Int]()
    val rseeds = Range(0, parts).map { i => (i, rand.nextInt) }.toArray
    //val rseeds = Range(0, parts).map { i => (i, (for(i<-0 until nbatch) yield rand.nextInt)) }.toArray
    sc.parallelize(rseeds, parts).map {
      case (i, seed) =>
        if (i < fullNum) (i * (avg + 1), (i + 1) * (avg + 1), seed)
        else (i * avg + fullNum, (i + 1) * avg + fullNum, seed)
    }
  }

  def run(args: Array[String]): Unit = {
    val out = new PrintWriter("./Data_centers.log")
    //blank items: "NAN, ,NA,nothing,."
    println(args.length)
    println(args.mkString(","))
    if (args.length != 8) {
      System.err.println("ERROR: expected 8 args\n<spark master> <path of data> <number of partitions> <number of records> <number of features> <numerical table name> <num clusters> <cluster scatter>")
      System.exit(1)
    }
    var DATA_PATH = args(1)
    val nparts = args(2).toInt
    val nRecords = args(3).toInt
    val num_features = args(4).toInt
    val NumericalTable = args(5)
    val numcluster = args(6).toInt
    val clusterscatter = args(7).toDouble
    val globalRandom = new Random(23)


    val conf = new SparkConf().setMaster(args(0)).setAppName("Spectral Clustering Synthetic Data Generator")

    val sc = new SparkContext(conf)
    val nFeatures_br = sc.broadcast(num_features)
    val numcluster_br = sc.broadcast(numcluster)

    if(! DATA_PATH.endsWith("/")){
      println("WARNING: path may cause errors. This should be a folder with '/'. Attempting to fix.")
      DATA_PATH = DATA_PATH + "/"
    }

    try {
      ("hadoop fs -rmr " + DATA_PATH + NumericalTable).!
    }catch{
      case e: Exception => println("indicated path does not exist!")
    }

    val DataRange = getRangeRDD(sc, globalRandom, nRecords, nparts)


    val cluster_data = for (i <- 0 until numcluster) yield {
      val center = for (j <- 0 until num_features) yield {
        globalRandom.nextDouble()
      }
      center.toArray
    }


    val cluster_data_br = sc.broadcast(cluster_data)

    println(">>> Generating synthetic clustering points ......")
    DataRange.flatMap {
      case (startx, endx, rinit) =>
        val rand = new Random(rinit)
        for (i <- startx until endx) yield {
          val center_now = cluster_data_br.value(rand.nextInt(numcluster_br.value))
          val rand_data = for (j <- 0 until nFeatures_br.value) yield {
            var res = rand.nextDouble() * clusterscatter + center_now(j)
            if (res < 0) res = 0.0
            if (res > 1) res = 1.0
            res
          }
          /*val miss_entries = for (k <- 0 until (nFeatures.value * LostRatio).toInt) yield {
            rand.nextInt(nFeatures.value)
          }
          val rand_data = for (j <- 0 until nFeatures.value) yield {
            if (miss_entries.contains(j)){
              val blank_inx = rand.nextInt(br_blanklength.value)
              BlankItem(blank_inx)
            }
            else {
              j%3 match {
                case 0 => abs(rand.nextGaussian()).toString
                case 1 => abs(rand.nextGaussian() * 0.5 + rand.nextInt(5)).toString
                case 2 => abs(rand.nextDouble()).toString
              }
              //abs(rand.nextGaussian()).toString()
            }*/

          rand_data.mkString(",")
        }
    }.saveAsTextFile(DATA_PATH + NumericalTable)
    out.println("clustering centers are:")
    out.println(cluster_data.map(i => i.mkString(",")).mkString("\n"))
    out.close()
  }
  def main(args: Array[String]): Unit ={
    run(args)
  }


}
