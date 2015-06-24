package com.Intel.bigDS.hash

import org.apache.spark.mllib.linalg.{DenseVector,SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ListBuffer
import org.apache.spark.SparkContext._
import java.io.PrintWriter

class LSH(zdata : RDD[(Vector, Long)], hyperplanes : Array[Int], thresholds : Array[Double], M : Int) extends Serializable {

  /** run LSH using the constructor parameters */
  def run(): LSHModel = {

    val sc = zdata.sparkContext

    val model = new LSHModel(hyperplanes, thresholds, M)
    val model_br = sc.broadcast(model)


    val signatures = zdata.map(v => {
      (v._2, model_br.value.hashFunctions.map(h => h._1.RandomProj(v._1) << h._2).sum)
    }).collect()


    val distinctsig = signatures.map(_._2).distinct
    val T = distinctsig.size
    var bucket_index = 0
    val bucket_alloc = new Array[Int](T).map(i => -1)
    for (i <- 0 until T) {
      if (bucket_alloc(i) == -1) {
        bucket_alloc(i) = bucket_index
        for (j <- i + 1 until T) {
          if (bucket_alloc(j) == -1) {
            //O(1) algorithm to judge if two bit numbers differ by 1 bit. simjudge==0 infers that it's true.
            //And thus two signatures are put into the same bucket.
            val simjudge = (distinctsig(i) ^ distinctsig(j)) & ((distinctsig(i) ^ distinctsig(j)) - 1)
            if (simjudge == 0) {
              bucket_alloc(j) = bucket_index
            }
          }
        }
        bucket_index = bucket_index + 1
      }
    }

    val sigtobuck = distinctsig.zip(bucket_alloc).toMap

    val vectobuck = signatures.map(i => {
      (i._1.toLong, sigtobuck(i._2))
    })

    model.vector_cluster = sc.parallelize(vectobuck, zdata.partitions.size)
    model.clusters = zdata.map(_.swap).join(model.vector_cluster).map(i => (i._2._2, (i._2._1, i._1))).groupByKey()

    model

  }
}


