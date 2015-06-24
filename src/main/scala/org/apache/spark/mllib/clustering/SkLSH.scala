package org.apache.spark.mllib.clustering


import java.io.PrintWriter

import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.linalg.distributed.{PatchedRowMatrix, RowMatrix}
import scala.collection.mutable.PriorityQueue
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import scala.collection.mutable.PriorityQueue
import scala.util.Random
import java.io.PrintWriter


/**
 * Spectral K-means implementation.
 * inner block similarity sparsity has been added, according to
 * Chen W Y, Song Y, Bai H, et al.
 * Parallel spectral clustering in distributed systems[J].
 * Pattern Analysis and Machine Intelligence, IEEE Transactions on, 2011, 33(3): 568-586.
 *
 * The algorithm implementation is mainly based on this paper
 * Hefeeda M, Gao F, Abd-Almageed W.
 * Distributed approximate spectral clustering for large-scale datasets[C]
 * Proceedings of the 21st international symposium on High-Performance Parallel and Distributed Computing. ACM, 2012: 223-234.
 */


class SkLSH(private var k:Int,
            private var numDims:Int,
            private var sigma: Double,
            private var subpartial: Double) extends KMeans with Serializable with Logging {

  implicit object TupleOrdering extends Ordering[(Long, Double)] with Serializable {
    def compare(a: (Long, Double), b: (Long, Double)) = if (a._2 < b._2) -1 else if (a._2 > b._2) 1 else 0
  }

  def this() = this(3, -1, 1.0, 1.0)

  def setDims(Dim: Int): this.type = {
    this.numDims = Dim
    this
  }

  def setk(k: Int): this.type = {
    this.k = k
    this
  }

  def setSigma(sigma:Double): this.type = {
    this.sigma = sigma
    this
  }

  def setSubPartial(subpartial: Double): this.type = {
    this.subpartial = subpartial
    this
  }

  def SpectralDimReduction(data: RDD[Vector], nParts: Int): RDD[(Vector, Vector)] = {
    val sc = data.sparkContext
    if (numDims < 0) numDims = data.count().toInt
    val parseddata = data.zipWithIndex().cache()

    val summary = Statistics.colStats(data)
    val colmax = summary.max.toArray
    val colmin = summary.min.toArray
    val k_br = sc.broadcast(k)

   val M = math.max((math.log(numDims)/math.log(2)).ceil.toInt,20)

    val colspan = colmax.zip(colmin).map(i => math.abs(i._1-i._2))
    val spansum = colspan.sum
    val colpossibility = colspan.map(i => i / spansum)
    val colinterval = colspan.map(i => i / 20)

    var incrementalvalue = 0.0
    val colincremental = for (i <- colpossibility) yield {
      incrementalvalue = incrementalvalue + i
      incrementalvalue
    }

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

    val colviewnum = hyperplanes.length
    val colmin_view = for (i <- hyperplanes) yield {colmin(i)}
    val colmax_view = for (i <- hyperplanes) yield {colmax(i)}

    val colinterval_view = for (i <- hyperplanes) yield {colinterval(i)}

    val colmin_br = sc.broadcast(colmin_view)
    val colinterval_br = sc.broadcast(colinterval_view)
    val hyperplanes_br = sc.broadcast(hyperplanes)

    val datawithbank = data.map(i => for (j <- hyperplanes_br.value) yield {i(j)}).map(i => i.toArray.zipWithIndex).map(i => i.map(j => {
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

    val databanksummary = datawithbank.treeAggregate(Array.ofDim[Int](colviewnum, 20))(
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

    val minbin = databanksummary.map(i => {
      val min_index = i.zipWithIndex.filter(j => j._1 == i.min).map(_._2)
      min_index(min_index.length / 2)
    })

    val thresholds = colmin_view.zip(colinterval_view).zip(minbin).map{case(((min,interval), bin)) => min + bin * interval}

    val lsh = new com.Intel.bigDS.hash.LSH(parseddata, hyperplanes, thresholds, M)
    val model = lsh.run()
    val subpartial_br = sc.broadcast(subpartial)
    val numDims_br = sc.broadcast(numDims)
    val sigma_br = sc.broadcast(sigma)
    val simiData = model.clusters.flatMap{i => {
      //similarity matrix sparsification. Maintain a priority queue for each data instance to retain only t nearest similarities
      val res = for (m <- i._2) yield {
        val queue = new PriorityQueue[(Long, Double)]
        val t = (subpartial_br.value * numDims_br.value).toInt
        for (n <- i._2.toArray) {
          val dis = (n._2, Vectors.sqdist(m._1, n._1))
          if (queue.size < t) {
            queue.enqueue(dis)
          }
          else if (queue.size >= t) {
            if (dis._2 < queue.head._2) {
              queue.dequeue()
              queue.enqueue(dis)
            }
          }
        }
        (m._2, queue.toArray)
      }
      //after conducting t-nearest similarity, the similairty matrix is not symmetric. So make it symmetric
      res.flatMap{case (row, cols) => for (i <- cols) yield { (row, (i._1, i._2))}}.flatMap{ case (row, (col, value)) => List((row, (col, value)), (col, (row, value)))}.groupBy(_._1)
        .map(i => {
        //avoid duplicates
        val sparse_data = i._2.map(_._2).toMap.toArray.map(i => (i._1, math.exp(-(i._2 * i._2) / (sigma_br.value * sigma_br.value))))
        (i._1, new SparseVector(numDims_br.value, sparse_data.map(_._1.toInt), sparse_data.map(_._2)))
      })
    }}.cache()

    val D_diag = simiData.map{case(index, content) => (index,math.pow(content.values.sum, 0.5))}.collect.toMap
    val D_diag_br = sc.broadcast(D_diag)
    //calculating Laplacian matrix. According to Apache Mahout, the implementation is not using L=I-D{-1/2}SD{-1/2} as the paper indicates
    //It uses L=D{1/2}SD{1/2}. I still dont know why. But it works:)
    val L = simiData.mapPartitions{ iter =>
      val D = D_diag_br.value
      iter.map{case(index, content) => {
        val indices = content.indices
        val values = content.values
        val Dim = content.size
        val num = values.length
        val newvalues = for(i <- 0 until num) yield{
          if (indices(i)==index) D(index)*values(i)*D(indices(i))
          else D(index)*values(i)*D(indices(i))
        }
        (index, new SparseVector(Dim, indices, newvalues.toArray))
      }
      }
    }

    val mat0 = new RowMatrix(L.map(i => i._2), numDims, numDims)
    val mat1 = L.map(i => i._1).zip(mat0.computeSVD(k, true).U.rows)
    def isallzero(input:Array[Double]): Boolean = {
      var flag:Boolean = true
      input.foreach(i => if(i!=0.0) flag=false)
      flag
    }
    //compute normalized matrix u_0
    val mat = mat1.mapPartitions { iter =>
      val k_0 = k_br.value
      iter.map { i => {
        if (isallzero(i._2.toArray)) {
          (i._1, Vectors.dense(Array.fill[Double](k_0)(1.0)))
        }
        else {
          val norm = Vectors.norm(i._2, 2)
          (i._1, Vectors.dense(i._2.toArray.map(j => j / norm)))
        }
      }
      }
    }
    parseddata.map(i => (i._2, i._1)).join(mat).map(_._2)
  }

  override def run(data:RDD[Vector]): SpectralKMeansModel = {
    val reduced_k = SpectralDimReduction(data, data.partitions.length)
    val data_to_cluster = reduced_k.map(_._2)
    //val kmeans_model = super.setK(k).run(data_to_cluster)
    val kmeans_model = super.run(data_to_cluster)
    new SpectralKMeansModel(kmeans_model.clusterCenters, reduced_k, data.partitions.length)
  }
}


object SkLSH extends Serializable {
  def train(
             data: RDD[Vector],
             k: Int,
             Dim: Int,
             sigma: Double,
             subpartial: Double,
             maxIterations: Int,
             runs: Int,
             initializationMode: String,
             seed: Long
  ): SpectralKMeansModel = {
    new SkLSH()
      .setK(k)
      .setk(k)
      .setMaxIterations(maxIterations)
      .setRuns(runs)
      .setSigma(sigma)
      .setSubPartial(subpartial)
      .setInitializationMode(initializationMode)
      .setSeed(seed)
      .setDims(Dim)
      .run(data)
  }
}



