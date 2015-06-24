package org.apache.spark.mllib.clustering


import java.io.PrintWriter

import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.linalg.distributed.{RowMatrix, PatchedRowMatrix, IndexedRowMatrix, IndexedRow}
import scala.collection.mutable.PriorityQueue
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import scala.collection.mutable.Seq
import scala.collection.mutable.HashMap
import org.apache.spark.mllib.linalg.BLAS.dot
import scala.collection.mutable.ArrayBuffer


/**
 * Spectral K-means implementation.
 * Based on this paper:
 * Chen W Y, Song Y, Bai H, et al.
 * Parallel spectral clustering in distributed systems[J].
 * Pattern Analysis and Machine Intelligence, IEEE Transactions on, 2011, 33(3): 568-586.
 * However, this implementation is not stable. Often, breeze or common math package would throw exception
 * that says the matrix is not positive semidefinate. I still cant figure that out.
 */

class SpectralKMeans(private var k:Int,
                     private var numDims:Int,
                     private var sparsity: Double) extends KMeans with Serializable with Logging {

  implicit object TupleOrdering extends Ordering[(Long, Double)] with Serializable {
    def compare(a: (Long, Double), b: (Long, Double)) = if (a._2 < b._2) -1 else if (a._2 > b._2) 1 else 0
  }

  def this() = this(2, -1, 0.1)


  def setk(k: Int): this.type = {
    this.k = k
    this
  }

  def setDims(Dim: Int): this.type = {
    this.numDims = Dim
    this
  }

  def setSparsity(p: Double): this.type = {
    this.sparsity = p
    this
  }

  /**
   *
   * @param data
   * @return RDD[(Vector, Vector)] consists of origianl vector and dimensionaly reduced vector
   */
  def SpectralDimReduction(data: RDD[Vector], nParts: Int): RDD[(Vector, Vector)] = {
    @transient val sc = data.sparkContext
    if (numDims < 0) numDims = data.count().toInt
    val DataWithIndex = data.zipWithIndex().map(i => (i._2, new VectorWithNorm(i._1))).partitionBy(new HashPartitioner(nParts)).cache()
    //retrieve & broadcast data RDD partition by partition to get mutual distance matrix
    //create a heap for sparsity creation
    var tempRDD = DataWithIndex.map(i => (i._1, (i._2,  new PriorityQueue[(Long, Double)])))
    val parts = DataWithIndex.partitions
    val t_value = (numDims * sparsity).ceil.toInt
    val k_br = sc.broadcast(k)
    //val t_br = sc.broadcast(t_value)
    for (p <- parts) {
      val t_br = sc.broadcast(t_value)//Why I have to put this inside the for loop? If I put it outside, an unserializable error will be invoked.
      val idx = p.index
      val partRdd = DataWithIndex.mapPartitionsWithIndex((index, element) =>
        if (index == idx) element else Iterator(), true)
      val PartialData = partRdd.collect
      //utilize sparsity parameter
      //val help = new forT((numDims * sparsity).ceil.toInt)

      val br_pd = sc.broadcast(PartialData)
      //similarity matrix and nearest neighbors (building heaps)
      tempRDD = tempRDD.mapPartitions { iter => {
        val pd = br_pd.value
        val t = t_br.value
        iter.map { case (index, (vector, queue)) => {
          for (j <- pd) {
            j match {
              case (index2, vector2) => {
                 val dist = SpectralKMeans.fastSquaredDistance(vector, vector2)
                if (queue.size < t) {
                  queue.enqueue((index2, dist))
                }
                else if (queue.size >= t) {
                  if (dist < queue.head._2) {
                    queue.dequeue()
                    queue.enqueue((index2, dist))
                  }
                }
              }
              case _ => throw new IllegalArgumentException("")
            }
          }
          (index, (vector, queue))
        }
        }
      }
      }
    }


    val disData = tempRDD.map { case (index, (dimdata, queue)) => (index, queue.toArray)}
      .flatMap { case (row, col_dis_array) => {
      val parse = ArrayBuffer[(Long, (Long, Double))]()
      for (m <- col_dis_array) {
        parse ++= ArrayBuffer((row, (m._1, m._2)), (m._1, (row, m._2)))
      }
      parse.toSeq
      }
    }
    val numDims_br = sc.broadcast(numDims)
      //generate symmetric matrix
    val disData_sym = disData.groupByKey()
    .map { case (row, iter) => {
    val DistFac = new HashMap[Int, Double]
    val numDims = numDims_br.value
    iter.map(j => {
      if (j._1 >= numDims) throw new IllegalArgumentException("Given dimension incorrect! Set numDim=-1 to auto detect.")
      DistFac.getOrElseUpdate(j._1.toInt, j._2)
    })
    val ave_row = iter.map(a => (a._2, 1)).reduce((a, b) => (a._1 + b._1, a._2 + b._2))
    val avg = ave_row._1.toDouble / ave_row._2
    val sortedDistFac = DistFac.toSeq.sortBy(_._1).toMap//sparse vector's indices must be increasingly ordered
    (row, sortedDistFac, avg)
    }
    }
    val avg_col_br = sc.broadcast(disData_sym.map(i => (i._1,i._3)).collect)

    //compute similiarities
   val simiData = disData_sym.mapPartitions { iter => {
      val col_avg = avg_col_br.value.toMap
      iter.map { i =>
        val simi = i._2.map(j => (j._1, math.exp((-j._2 * j._2) / (col_avg(i._1) * col_avg(j._1)))))
        //new IndexedRow(i._1.toLong, Vectors.sparse(num_dim, simi.keys.toArray, simi.values.toArray))
        (i._1, simi)
      }
    }
    }
    //val out = new PrintWriter("/home/yaochunnan/Intel-BDT/SpectralKMeans/ref/metadata.txt")
    //out.println("Row ID: Feature1,Feature2,Feature3,Feature4,Feature5,Feature6")
    //out.print(simiData.map(i => i._1 + ":" + " " + Vectors.dense(Vectors.sparse(numDims, i._2.keys.toArray, i._2.values.toArray).toArray).toArray.mkString(",")).collect.mkString("\n"))

    //Construct the diagonal matrix D (represented as a vector), and broadcast it
    val D_diag = simiData.mapValues(i => math.pow(i.values.sum, 0.5)).collect.toMap
    val D_br = sc.broadcast(D_diag)

    val dims_br = sc.broadcast(numDims)
    val L = simiData.mapPartitions { iter => {
      val D_exp = D_br.value
      val dims = dims_br.value
      iter.map { case (row_idx, simimap) => {
        val res = simimap.map { case (col_idx, value) => {
          var L_elements = 0.0
          val DSD = D_exp(row_idx) * value * D_exp(col_idx)
          if(col_idx == row_idx) L_elements =DSD
          else L_elements = DSD
          (col_idx, L_elements)
          }
        }
        (row_idx.toInt, Vectors.sparse(dims, res.keys.toArray, res.values.toArray))
        }
      }
      }
    }.map(i => (i._1.toLong, i._2))

   // out.println("Laplacian Matrix")
   // out.println("Row ID: Feature1,Feature2,Feature3,Feature4,Feature5,Feature6")
   // out.print(L.map(i => i._1 + ":" + " " + Vectors.dense(i._2.toArray).toArray.mkString(",")).collect.mkString("\n"))

    //System.exit(1)

    //val mat0 = new PatchedRowMatrix(sc, L.partitions.length, L.map(i => i._2), numDims, L.map(i => i._2).first.size)
    val mat0 = new RowMatrix(L.map(i => i._2), numDims, numDims)
    //parallel eigensolver
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

    DataWithIndex.map(i => (i._1, i._2.vector)).join(mat).map(_._2)
  }

  override def run(data:RDD[Vector]): SpectralKMeansModel = {
    val reduced_k = SpectralDimReduction(data, data.partitions.length)
    val data_to_cluster = reduced_k.map(_._2)
    val kmeans_model = super.run(data_to_cluster)
    new SpectralKMeansModel(kmeans_model.clusterCenters, reduced_k, data.partitions.length)
  }
}

object SpectralKMeans extends Serializable{
  def train(
             data: RDD[Vector],
             k: Int,
             Dim: Int,
             sparsity: Double,
             maxIterations: Int,
             runs: Int,
             initializationMode: String,
             seed: Long): SpectralKMeansModel = {
   new SpectralKMeans()
      .setK(k)
      .setk(k)
      .setMaxIterations(maxIterations)
      .setRuns(runs)
      .setInitializationMode(initializationMode)
      .setSeed(seed)
      .setSparsity(sparsity)
      .setDims(Dim)
      .run(data)
  }

  private[clustering] lazy val EPSILON = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }

  private[clustering] def fastSquaredDistance(
                                               v1: VectorWithNorm,
                                               v2: VectorWithNorm): Double = {
    fastSquaredDistance(v1.vector, v1.norm, v2.vector, v2.norm)
  }

  private[clustering] def fastSquaredDistance(
                                          v1: Vector,
                                          norm1: Double,
                                          v2: Vector,
                                          norm2: Double,
                                          precision: Double = 1e-6): Double = {
    val n = v1.size
    require(v2.size == n)
    require(norm1 >= 0.0 && norm2 >= 0.0)
    val sumSquaredNorm = norm1 * norm1 + norm2 * norm2
    val normDiff = norm1 - norm2
    var sqDist = 0.0
    /*
     * The relative error is
     * <pre>
     * EPSILON * ( \|a\|_2^2 + \|b\\_2^2 + 2 |a^T b|) / ( \|a - b\|_2^2 ),
     * </pre>
     * which is bounded by
     * <pre>
     * 2.0 * EPSILON * ( \|a\|_2^2 + \|b\|_2^2 ) / ( (\|a\|_2 - \|b\|_2)^2 ).
     * </pre>
     * The bound doesn't need the inner product, so we can use it as a sufficient condition to
     * check quickly whether the inner product approach is accurate.
     */
    val precisionBound1 = 2.0 * EPSILON * sumSquaredNorm / (normDiff * normDiff + EPSILON)
    if (precisionBound1 < precision) {
      sqDist = sumSquaredNorm - 2.0 * dot(v1, v2)
    } else if (v1.isInstanceOf[SparseVector] || v2.isInstanceOf[SparseVector]) {
      val dotValue = dot(v1, v2)
      sqDist = math.max(sumSquaredNorm - 2.0 * dotValue, 0.0)
      val precisionBound2 = EPSILON * (sumSquaredNorm + 2.0 * math.abs(dotValue)) /
        (sqDist + EPSILON)
      if (precisionBound2 > precision) {
        sqDist = Vectors.sqdist(v1, v2)
      }
    } else {
      sqDist = Vectors.sqdist(v1, v2)
    }
    sqDist
  }

}



