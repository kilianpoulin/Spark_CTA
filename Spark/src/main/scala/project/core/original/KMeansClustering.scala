package project.core.original

import java.io.File
import java.lang.Math.{pow, sqrt}

import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec
import scala.util.Random
import org.apache.spark.mllib.linalg.Vector
import breeze.linalg.{sum, DenseVector => BVector}
import org.apache.spark.SparkContext

import scala.collection.mutable

  case class VData (v: BVector[Double]) {
  // for the following functions, v is the vector and c the centroid
  def distanceTo(c: Vector) = sqrt(pow(v(0) - c(0), 2) + pow(v(1) - c(1), 2) + pow(v(2) - c(2), 2))

  def sum(c: Vector) = VData(BVector(v(0) + c(0), v(1) + c(1), v(2) + c(3)))

  def divideBy(number: Double) = VData(BVector(v(0) / number, v(1) / number, v(2) / number))

}

class KMeansClustering (
                        sc: SparkContext,
                         private var k: Int,
                         private var maxIterations: Int,
                         //private var initializationSteps: Int,
                         //private var epsilon: Double,
                         private var tensorDim: Int) extends Serializable {



  def train(
             data: RDD[Vector]
             //initializationMode: String,
             //seed: Long
           ) = {
    //new KMeansClustering(k, maxIterations, tensorDim)
      //.setSeed(seed)
      run(data)
  }

  /*
  def CalcPartialDist(r: RDD[Vector]) : RDD[(mutable.ArraySeq[Double], RDD[Vector])] = {

  }*/


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

    // Step 1 : create k clusters with random values as centroids

    createRandomCentroids(data)
    /*val clusters = buildClusters(data, createRandomCentroids(data))
    clusters.foreach({
      case (centroid, members) =>
        members.foreach({ member => println(s"Centroid: $centroid Member: $member") })
    })*/

    // Step 2 : Calculate partial distances
    //CalcPartialDist(data)

  }

  // should return Map[Vector, RDD[Vector]]
  def createRandomCentroids(data: RDD[Vector]) = {
  //: scala.collection.Map[Vector, RDD[Vector]] = {
    val randomIndices = collection.mutable.HashSet[Int]()
    val random = new Random()
    while (randomIndices.size < k) {
      // choosing on of the vectors in the dataset to become one cluster centroid
      // so we will choose k vectors in the dataset
      randomIndices += random.nextInt(data.count().toInt)
    }
    val tmpRDD = sc.parallelize(Vector(null))

    println("nb vect = " +data.count())

    val myvect = data.zipWithIndex.filter({case (_, index) => randomIndices.contains(index.toInt)}).map({ case (vect, _) => (vect, tmpRDD)}).map(s => s._1).take(1)
    println("sum : " + sum(myvect(0).toArray))

    // now we find these vectors in the dataset using their ID, previously randomly selected
    // we add another element to these vectors : which will be the vector of the cluster
    // thus, each cluster is an RDD / map containing a pair (Vector : centroid, Vector: cluster values)
    //
    /*
    data
      .zipWithIndex
      .filter({ case (_, index) => randomIndices.contains(index.toInt) })
      .map({ case (point, _) => (point, tmpRDD) })
      .collectAsMap().foreach(println)
    data
      .zipWithIndex
      .filter({ case (_, index) => randomIndices.contains(index.toInt) })
      .map({ case (point, _) => (point, tmpRDD) })
      .collectAsMap()*/
  }
  /*

  @tailrec
  def buildClusters(data: RDD[Vector], prevClusters: scala.collection.Map[Vector, RDD[VData]]): Map[Vector, RDD[VData]] = {
    val nextClusters = data.map({ vect =>
      val byDistanceToPoint = new Ordering[VData] {
        override def compare(v1: VData, v2: VData) = v1.distanceTo(vect) compareTo v2.distanceTo(vect)
      }

      (vect, prevClusters.keys min byDistanceToPoint)
    }).groupBy({ case (_, centroid) => centroid })
      .map({ case (centroid, pointsToCentroids) =>
        val points = pointsToCentroids.map({ case (point, _) => point })
        (centroid, points)
      })

    prevClusters
    /*

    if (prevClusters != nextClusters) {
      val nextClustersWithBetterCentroids = nextClusters.map({
        case (centroid, members) =>
          val (sum, count) = members.foldLeft((Vector(0, 0, 0), (0))({ case ((acc, c), curr) => (acc sum curr.t, c + 1) })
          (sum divideBy count, members)
      })

      buildClusters(data, nextClustersWithBetterCentroids)
    } else {
      prevClusters
    }*/
  }*/

}