
package org.apache.spark.mllib.clustering

import java.io.PrintWriter

import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.linalg.distributed.{PatchedRowMatrix, IndexedRowMatrix, IndexedRow}
import scala.collection.mutable.PriorityQueue
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import scala.collection.mutable.Seq
import scala.collection.mutable.HashMap
import org.apache.spark.mllib.linalg.BLAS.dot
import scala.collection.mutable.ArrayBuffer
import breeze.linalg.DenseMatrix
import org.apache.commons.math3.linear._



/**
 * Spectral K-means implementation using the Nystrom method.
 * Based on this paper:
 * Chen W Y, Song Y, Bai H, et al.
 * Parallel spectral clustering in distributed systems[J].
 * Pattern Analysis and Machine Intelligence, IEEE Transactions on, 2011, 33(3): 568-586.
 * However, this implementation is not stable. Often, breeze or common math package would throw exception
 * that says the matrix is not positive semidefinate. I still cant figure that out.
 */

class SkNystrom(private var k:Int,
                     private var numDims:Int,
                     private var partial: Double,
                     private var sigma: Double) extends KMeans with Serializable with Logging {

  def this() = this(2, -1, 0.2, 1.0)

  def setk(k: Int): this.type = {
    this.k = k
    this
  }

  def setDims(Dim: Int): this.type = {
    this.numDims = Dim
    this
  }

  def setPartial(p: Double): this.type = {
    this.partial = p
    this
  }

  def setSigma(sigma:Double): this.type = {
    this.sigma = sigma
    this
  }

  /**
   *
   * @param data_row
   * @param nParts
   * @return RDD[(Vector, Vector)] consists of origianl vector and dimensionaly reduced vector
   */
  def SpectralDimReduction(data_row: RDD[Vector], nParts: Int): RDD[(Vector, Vector)] = {
    @transient val sc = data_row.context

    val out = new PrintWriter("./watchout_Nystrom_inner.log")

    val data = data_row
    if (numDims < 0) numDims = data.count.toInt
    val DataWithIndex = data.zipWithIndex().map(i => (i._2, new VectorWithNorm(i._1))).partitionBy(new HashPartitioner(nParts))
    //val l = (partial * numDims).toInt + 1
    val SplitArray = DataWithIndex.randomSplit(Array(partial, 1 - partial), 3)
    val LocalVec = SplitArray(0).collect
    out.println("LocalVec selected randomly")
    out.println(LocalVec.map(i => i._1 + " : " + i._2.vector.toArray.mkString(",")).mkString("\n"))

    val RddVec = SplitArray(1)

    out.println("RDD vectors")
    out.println(RddVec.map(i => i._1 + " : " + i._2.vector.toArray.mkString(",")).collect.mkString("\n"))

    val br_LocalVec = sc.broadcast(LocalVec)
    val l = LocalVec.length

    val l_br = sc.broadcast(l)
    val k_br = sc.broadcast(k)
    require(l > k && partial > 0 && partial < 1, s"matrix A too small. expect much larger than $k, in fact l=$l")
    val MatA_dis = for (i <- 0 until l) yield {
      val row_array = new Array[Double](l)
      for (k <- 0 until l) {
        val dis = SkNystrom.fastSquaredDistance(LocalVec(i)._2, LocalVec(k)._2)
        row_array(k) = dis
      }
      (LocalVec(i)._1, row_array)
    }

    //RDD[(col,Array[Double])]
    val MatBRdd_dis = RddVec.mapPartitions { iter => {
      val lv = br_LocalVec.value
      iter.map { vector => {
        val column_l = for (i <- lv) yield {
          val dis = SkNystrom.fastSquaredDistance(vector._2, i._2)
         /* if (dis==0.0) 0.0001
          else dis*/
          dis
        }
        (vector._1, column_l)
        }
      }
      }
    }

    //val avg_row_br = sc.broadcast(avg_row)
    val numDims_br = sc.broadcast(numDims)
    val sigma_br = sc.broadcast(sigma)

    val MatA_indexed = MatA_dis.map(i => {
      if (i._1 >= numDims) throw new IllegalArgumentException("Given dimension incorrect! Set numDim=-1 to auto detect.")
      (i._1, i._2.map{ j =>
        val res = math.exp(- (j * j) / (2 * sigma * sigma))
      /*  if (res==1.0) {
         0.99
        }
        else res*/
        res
      })
    })
    val MatBRdd_indexed = MatBRdd_dis.mapPartitions{iter => {
      val B1_dis = sigma_br.value
      val numDims = numDims_br.value
      iter.map{i => {
        if (i._1 >= numDims) throw new IllegalArgumentException("Given dimension incorrect! Set numDim=-1 to auto detect.")
        (i._1, i._2.map{j =>
          val res = math.exp(- (j * j) / (2 * B1_dis * B1_dis))
          /*if (res==1.0) {
            0.99
          }
          else res*/
          res
        })
      }}
    }}


    out.println("MatA_indexed")
    out.println(MatA_indexed.map(i => i._1 + " : " + i._2.mkString(",")).mkString("\n"))
    out.println("MatB_indexed")
    out.println(MatBRdd_indexed.map(i => i._1 + " : " + i._2.mkString(",")).collect.mkString("\n"))

    val MatA = MatA_indexed.map(_._2)
    val MatAt = Array.ofDim[Double](l, l)
    for (i <- 0 until l) {
      for (j <- 0 until l) {
        MatAt(i)(j) = MatA(j)(i)
      }
    }
    val MatBRdd = MatBRdd_indexed.map(_._2)
    val a = MatA.map(_.sum)
    val b1 = MatBRdd.treeAggregate(new Array[Double](l))(
      seqOp = (U, r) => {
        U.zip(r).map{i => i._1+i._2}
      },
      combOp = (U1, U2) => {
        U1.zip(U2).map(i => i._1+i._2)
      }
    )

    val b2 = MatBRdd.map(i => i.sum)
    val MatA_rev = breeze.linalg.inv(new DenseMatrix(l, l, MatAt.flatMap(i => i))).toArray
    val br_MatA_rev = sc.broadcast(MatA_rev)
    val br_b1 = sc.broadcast(b1)
    val b3 = MatBRdd.mapPartitions{iter => {
      val MatA_r = br_MatA_rev.value
      val B1 = br_b1.value
      val l = l_br.value
      iter.map{i => {
        var outer:Double=0.0
        for (n <- 0 until l) {
          var inner:Double=0.0
          for (p <- 0 until l) {
            inner += i(p) * MatA_r(n*l+p)
          }
          outer += inner * B1(n)
        }
        outer
      }
      }
    }
    }

    val D_1 = a.zip(b1).map(i => i._1+i._2)
    val D_1_half = D_1.map(i => math.pow(i, -0.5))


    val D_1_half_br = sc.broadcast(D_1_half)
    val D_2 = b2.zip(b3).map(i => {
      val res = i._1+i._2
      if (res==0.0) res
      else res
    })
    val A_0 = Array.ofDim[Double](l, l)
    for (i <- 0 until l) {
      for (j <- 0 until l) {
        A_0(i)(j) = D_1_half(i) * MatA(i)(j) * D_1_half(j)
      }
    }


    val B_00 = MatBRdd.mapPartitions{iter => {
      val d_1_half = D_1_half_br.value
      val L = l_br.value
      iter.map{i => {
        for(j <- 0 until L) yield {
          d_1_half(j) * i(j)
        }
      }}
    }}
    val B_0 = B_00.zip(D_2).map{case(a,b) => a.map(_*math.pow(b, -0.5))}.map(i => i.toArray)



    //[A_0 B_0t]
    val A_0_B_0 = sc.parallelize(A_0.zip(MatA_indexed.map(_._1))).union(B_0.zip(MatBRdd_indexed.map(_._1)))


    val mat:RealMatrix = new Array2DRowRealMatrix(A_0, false)
    val eigen = new EigenDecomposition(mat)
    val A_half = eigen.getSquareRoot.getData
    val A_half_br = sc.broadcast(A_half)
    //A_0{-1/2}*B_0
    val A_1 = MatBRdd.mapPartitions{iter => {
      val a_half = A_half_br.value
      val L = l_br.value
      iter.map{i => {
        for(j <- 0 until L) yield {
          a_half(j).zip(i).map{case(a,b) => a*b}.sum
        }
      }}
    }}
    //A_0{-1/2}*B_0*B_0t
   // val A_1_Bt = Array.ofDim[Double](l, l)
    val A_1_Bt = MatBRdd.zip(A_1).mapPartitions{iter =>
      val l = l_br.value
      iter.map{i =>
        val col = Array.ofDim[Double](l, l)
        for (m <-0 until l) {
          for (n <- 0 until l) {
            col(m)(n) = i._2(m) * i._1(n)
          }
        }
        col
      }
    }.treeAggregate(Array.ofDim[Double](l,l))(
      seqOp = (U, r) => {
        U.zip(r).map(i => i._1.zip(i._2).map(i => i._1 + i._2))
      },
      combOp = (U1, U2) => {
        U1.zip(U2).map(i => i._1.zip(i._2).map(i => i._1 + i._2))
      }
    )

    val R = Array.ofDim[Double](l, l)

    for (i <- 0 until l) {
      for (j <- 0 until l) {
        R(i)(j) = A_1_Bt(i).zip(A_half(j)).map{case(a,b)=> a*b}.sum + A_0(i)(j)
      }
    }


    val mat2:RealMatrix = new Array2DRowRealMatrix(R, false)
    val eigen2 = new EigenDecomposition(mat2)
    val Ur = eigen2.getV.getData.map(_.take(k))
    val Ar = eigen2.getD.getData.take(k).map(_.take(k)).map(i => i.map(j => math.pow(j, -0.5)))
    val Ar_br = sc.broadcast(Ar)
    val Ur_br = sc.broadcast(Ur)



    val v_0 = A_0_B_0.mapPartitions{iter =>
      val ar = Ar_br.value
      val ur = Ur_br.value
      val a_half = A_half_br.value
      val k_0 = k_br.value
      iter.map{ i => {
        val value = for (s <- 0 until k_0) yield{
          var outer = 0.0
          for (m <- 0 until l) {
            var inner = 0.0
            for (n <- 0 until l) {
              inner += i._1(n) * a_half(n)(m)
            }
            outer = inner * ur(m)(s)
          }
          outer*ar(s)(s)
        }
        (i._2, value.toArray)
      }
      }
    }

//In case that all the entries of a line in v_0 are zeros. Thus the normalization would generate NAN entries to disable KMeans later on.
    def isallzero(input:Array[Double]): Boolean = {
      var flag:Boolean = true
      input.foreach(i => if(i!=0.0) flag=false)
      flag
    }
    //compute normalized matrix u_0
    val u_0 = v_0.mapPartitions { iter =>
      val k_0 = k_br.value
      iter.map { i => {
        if (isallzero(i._2)) {
          (i._1, Vectors.dense(Array.fill[Double](k_0)(1.0)))
        }
        else {
          val vec = Vectors.dense(i._2)
          val norm = Vectors.norm(vec, 2)
          (i._1, Vectors.dense(i._2.map(j => j / norm)))
        }
      }
      }
    }


    val finalres = DataWithIndex.map(i => (i._1, i._2.vector)).join(u_0).map(_._2)
    finalres
  }

  override def run(data:RDD[Vector]): SpectralKMeansModel = {
    val reduced_k = SpectralDimReduction(data, data.partitions.length)
   // println(reduced_k.count)
   // println("length of proj2")
   // System.exit(1)
    val data_to_cluster = reduced_k.map(_._2)
    //val kmeans_model = super.setK(k).run(data_to_cluster)
    val kmeans_model = super.run(data_to_cluster)
    new SpectralKMeansModel(kmeans_model.clusterCenters, reduced_k, data.partitions.length)
  }

}

object SkNystrom extends Serializable {
  def train(
             data: RDD[Vector],
             k: Int,
             Dim: Int,
             partial: Double,
             maxIterations: Int,
             runs: Int,
             initializationMode: String,
             seed: Long,
             sigma: Double): SpectralKMeansModel = {
    new SkNystrom()
      .setK(k)
      .setk(k)
      .setSigma(sigma)
      .setMaxIterations(maxIterations)
      .setRuns(runs)
      .setInitializationMode(initializationMode)
      .setSeed(seed)
      .setPartial(partial)
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
