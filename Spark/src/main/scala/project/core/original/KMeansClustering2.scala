package project.core.original

import java.io.File
import java.lang.Math.{pow, sqrt}

import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec
import scala.util.Random
import org.apache.spark.mllib.linalg.Vector

  case class VData (vect: Vector) {
  // for the following functions, v is the vector and c the centroid
  def distanceTo(vect2: Vector) = sqrt(pow(v(0) - c(0), 2) + pow(v(1) - c(1), 2) + pow(v(2) - c(2), 2))

  def sum(vect2: Vector) = VData(Vector(v(0) + c(0), v(1) + c(1), v(2) + c(3)))

  def divideBy(vect2: Vector, number: Double) = VData(v(0) / number, v(1) / number, v(2) / number)

}

class KMeansClustering (
                         private var k: Int,
                         private var maxIterations: Int,
                         private var initializationSteps: Int,
                         //private var epsilon: Double,
                         private var tensorDim: Int) extends Serializable {



  def train(
             data: RDD[Vector],
             k: Int,
             maxIterations: Int,
             initializationMode: String,
             seed: Long): KMeansModel = {
    new KMeansClustering().setK(k)
      .setSeed(seed)
      .run(data)
  }


  /**
  * Set the number of clusters to create (k).
    *
  * @note It is possible for fewer than k clusters to
  * be returned, for example, if there are fewer than k distinct points to cluster. Default: 2.
  */
  def setK(k: Int): this.type = {
    require(k > 0,
      s"Number of clusters must be positive but got ${k}")
    this.k = k
    this
  }

  def run(data: RDD[Vector]){
    val clusters = buildClusters(data, createRandomCentroids(data))
    clusters.foreach({
      case (centroid, members) =>
        members.foreach({ member => println(s"Centroid: $centroid Member: $member") })
    })
  }

  def createRandomCentroids(data: RDD[Vector]): Map[Vector, RDD[Vector]] = {
    val randomIndices = collection.mutable.HashSet[Int]()
    val random = new Random()
    while (randomIndices.size < k) {
      randomIndices += random.nextInt(data.count().toInt)
    }

    data
      .zipWithIndex
      .filter({ case (_, index) => randomIndices.contains(index.toInt) })
      .map({ case (point, _) => (point, Nil) })
  }

  @tailrec
  def buildClusters(data: RDD[Vector], prevClusters: Map[Vector, RDD[Vector]]): Map[Vector, RDD[VData]] = {
    val nextClusters = data.map({ vect =>
      val byDistanceToPoint = new Ordering[Vector] {
        override def compare(v1: VData, v2: VData) = v1.distanceTo(vect) compareTo v2.distanceTo(vect)
      }

      (vect, prevClusters.keys min byDistanceToPoint)
    }).groupBy({ case (_, centroid) => centroid })
      .map({ case (centroid, pointsToCentroids) =>
        val points = pointsToCentroids.map({ case (point, _) => point })
        (centroid, points)
      })

    if (prevClusters != nextClusters) {
      val nextClustersWithBetterCentroids = nextClusters.map({
        case (centroid, members) =>
          val (sum, count) = members.foldLeft((Vector(0, 0, 0), 0))({ case ((acc, c), curr) => (acc sum curr, c + 1) })
          (sum divideBy count, members)
      })

      buildClusters(data, nextClustersWithBetterCentroids)
    } else {
      prevClusters
    }
  }

}