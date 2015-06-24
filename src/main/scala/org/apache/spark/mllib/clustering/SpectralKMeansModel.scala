package org.apache.spark.mllib.clustering

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.HashPartitioner


class SpectralKMeansModel (override val clusterCenters: Array[Vector],
                           val pointProj:RDD[(Vector, Vector)],
                           val nParts:Int = 4)
                            extends KMeansModel(clusterCenters) with Serializable {
 /* val partitioner_res = new HashPartitioner(nParts)
  pointProj.partitionBy(partitioner_res).persist()
  override def predict(point:Vector): Int = {
    val point_to_find = pointProj.lookup(point).head
    KMeans.findClosest(clusterCentersWithNorm, new VectorWithNorm(point_to_find))._1
  }
//use partitioner_res to elevate the effeciency of Vector matching between RDDs
  override def predict(points: RDD[Vector]): RDD[Int] = {
    val Points = points.map(i => (i, 0)).partitionBy(partitioner_res)
    val centersWithNorm = clusterCentersWithNorm
    val bcCentersWithNorm = points.context.broadcast(centersWithNorm)
    val resultRDD = Points.join(pointProj).map{i => {
        KMeans.findClosest(bcCentersWithNorm.value, new VectorWithNorm(i._2._2))._1
      }
    }
    resultRDD
  }*/

  //println(pointProj_local.size)
  //println("query map number")
/*
  println(pointProj.map(i => i._1.toArray.mkString(",") + "   ||   " + i._2.toArray.mkString(",")).collect.mkString("\n"))
  println("point Proj")
  println(pointProj.count)
  println("pooint Proj length")
  System.exit(1)*/
  override def predict(point:Vector): Int = {
    val point_to_find = pointProj.lookup(point)
    if (point_to_find.isEmpty) {
      //println("point to query")
     //println(point.toArray.mkString(","))
     //System.exit(1)
      throw new IllegalArgumentException("Input data point not exist in original data set")
    }
    else KMeans.findClosest(clusterCentersWithNorm, new VectorWithNorm(point_to_find.head))._1
  }

  def predictall(): RDD[(Vector, Int)] = {
    //println(clusterCenters.mkString("\n"))
    //System.exit(1)
    pointProj.map{ case (vector, proj) =>
      val res = KMeans.findClosest(clusterCentersWithNorm, new VectorWithNorm(proj))
      (vector, res._1)
    }

    /*
    //val Points = points.map(i => (i, 0)).partitionBy(partitioner_res)
    val bcCentersWithNorm = points.context.broadcast(clusterCentersWithNorm)
    val bc_pointProj = points.context.broadcast(pointProj_local)
    val resultRDD = points.mapPartitions { iter => {
      val pointProj_value = bc_pointProj.value
      val clusterCenters = bcCentersWithNorm.value
      iter.map { i => {
        val point_to_find = pointProj_value.get(i)
        if (point_to_find.isEmpty) throw new IllegalArgumentException("Input data point not exist in original data set")
        else KMeans.findClosest(clusterCenters, new VectorWithNorm(point_to_find.get))._1
      }
      }
    }
    }
    resultRDD*/
  }


  private def clusterCentersWithNorm: Iterable[VectorWithNorm] =
    clusterCenters.map(new VectorWithNorm(_))
}