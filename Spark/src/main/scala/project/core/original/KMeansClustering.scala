package project.core.original

import java.io.File
import java.lang.Math.{pow, sqrt}
import java.util

import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.rdd.RDD

import scala.math.Ordering
import scala.annotation.tailrec
import scala.collection.{mutable => CM}
import org.apache.spark.mllib.linalg.{DenseMatrix => DMatrix}

import scala.util.Random
import scala.{Vector => Vect}
import breeze.linalg.{DenseMatrix, Vector}
import project.core.original.RunTucker.{tensorInfo, tensorRDD}
import project.core.original.Tensor.{MyPartitioner, localTensorUnfoldBlock}
import breeze.linalg.{DenseMatrix => BMatrix}

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
class KMeansClustering (
                         private var k: Int,
                         private var maxIterations: Int,
                         private var tensorDim: Int) extends Serializable {

  //
  // Creating implicit class to add function distanceTo to Vector
  //
  @transient case class VData(@transient v1: Vect[Double]) extends Serializable {
    @transient val distanceTo = (v2: Vect[Double]) => {
      val s: Array[Double] = v1.toArray
      val t: Array[Double] = v2.toArray
      sqrt((s zip t).map{ case (x,y) => pow(x - y, 2) }.sum)
    }
  }


  @transient implicit def distancecalc(@transient v: Vect[Double]): VData = VData(v)

  def train(
             data: RDD[ (CM.ArraySeq[Int], DMatrix) ]
           ) = {
      run(data)
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

  def run(data: RDD[ (CM.ArraySeq[Int], DMatrix) ]){

    // Step 1 : Unfold each block and make one partition for one block
      val newdata = data.map{ case(ids, mat) =>
        (ids, localTensorUnfoldBlock( mat, ids, 0,  tensorInfo.blockRank, tensorInfo.blockNum, tensorInfo.tensorRank))
      }.reduceByKey( new MyPartitioner( tensorInfo.blockNum ), ( a, b ) => a + b )

      println(" (3) Tensor unfolded along Dimension 1 : OK")
      //newdata.foreach(println)

    // Step 2 : create k clusters with random values as centroids data)
      var centroids = newdata.map{ case(ids, mat) => (ids, createRandomCentroids(mat))}
      //clusters.foreach{ case(x, s) => s.foreach(println)}
      println(" (4) " + k + " splitted clusters successfully initialized : OK")


    // Step 3 : calculate distance between partial vectors and each partial centroid vector
      var distances = centroids.join(newdata).map { case (ids, (centroids, values)) => (ids, getPartialDistances(ids, centroids, Tensor.blockTensorAsVectors(values))) }
      println(" (5) calculating distances between partial vectors and each partial centroid vector : OK")
      // display list of distances
      //distances.map{ case(id, values) => values }.flatMap(v => v).map{ case(vector, distances) => distances}.flatMap(a => a).collect().foreach(println)



    // Step 4 : build clusters by adding partial distances and selecting the shortest one
      // adding partial distances
      //distances.map{case(ids, _) => ids}.map{case(id) => (id, (id(0), id(2), id(3)))}.groupBy(_._2).foreach(println)
      //var vectIds = distances.map{case(ids, _) => ids}.map{case(id) => (id, (id(0), id(2), id(3)))}.groupBy(_._2).map{case (i, iter) => iter.map(_._1)}

    // do it manually with a var ?
      var vectIds = distances.map({ x => getVectorIds(x._1) }).collect()

    // or do it directy on distances RDD
    var fullVectors = distances.map{ x => (getVectorIds(x._1), x._2) }.groupBy(_._1)
    println("fullVectors = " + fullVectors.count() + " distances = " + distances.count())



    /*
    //var clusters = centroids.join(newdata).map{ case(ids, (centroids, values)) => ((ids, centroids), buildClusters(ids, centroids, values))}
    var clusters = centroids.join(newdata).map{ case(ids, (centroids, values)) => (ids, buildClusters(ids, centroids, values))}
    println("cluster count = " + clusters.count())

    /* test centroids
    var test = clusters.take(8).map{ case((id, centroid), values) => centroid}
    println(test(0)(0))
    */
   // var test = clusters.take(8).map{ case((id, centroid), values) => values}.collect

   var test = clusters.map{ case(id, values) => values}
    //test.toLocalIterator.foreach(arr => println(arr))
    test.collect().toList.foreach(println)
    println(" (5) " + k + " clusters built with partial vectors : OK")
    */
   // var clusters = centroids.map{ case()
    /*
    for(i <- 1 until data.count().toInt){
      // Unfold tensor along the Dimension 1
      mat = null
      ids = null
      mat = data.map(s => s._2).take(i)
      ids = data.map(s => s._1).take(i)
      //println(i + " " + ids(0) + "number = " + mat(0).numRows)
      //data.take(i).map(s => s._1).foreach(println)
      dataMatrix = null
      dataMatrix = localTensorUnfoldBlock( mat(0), ids(0), 0,  tensorInfo.blockRank, tensorInfo.blockNum, tensorInfo.tensorRank)
      println(" (3) Tensor unfolded along Dimension 1 : OK")
      //var clustersTmp = createRandomCentroids(ids(0), mat(0))
    }*/
    //clustersTmp.foreach(println)
    //clustersTmp.foreach(println)
    //val clusters = buildClusters(data, createRandomCentroids(data))

   // println("------------------------- Clustering result ----------------------")
    //clusters.filter({case(x, _) => x.toArray.sum > 0}).foreach(println)
   // clusters.map({ case(centroid, members) => members.size}).foreach(println)
   /*clusters.foreach({
      case (centroid, members) =>
        members.foreach({ member => println(s"Centroid: $centroid Member: $members") })
    })*/

    // Step 2 : Calculate partial distances
    //CalcPartialDist(data)

  }

  def getVectorIds(ids: CM.ArraySeq[Int]): List[CM.ArraySeq[Int]] ={
    var listIds = new Array[CM.ArraySeq[Int]](tensorInfo.blockNum(1))
    for(i <- 0 until tensorInfo.blockNum(1)){
      listIds(i) = ids.clone()
      listIds(i)(1) = i
    }

    listIds.toList
  }

  def getMatVect(data: BMatrix[Double], row: Int): Vect[Double] ={
    var tmpArray = new Array[Double](data.cols)
    for(i <- 0 until data.cols){
      tmpArray(i) = data.apply(row, i)
    }
    tmpArray.toVector
  }

  // should return scala.collection.Map[Vector, RDD[Vector]]
  // Array[Array[Vector]]
  def createRandomCentroids(data: BMatrix[Double]) = {


    val randomIndices = ListBuffer[Int]()
    val random = new Random()
    var tmp = 0
    var nbIt = 0
    while (randomIndices.size < k) {
      // choosing one of the vectors in the dataset to become one cluster centroid
      // so we will choose k vectors in the dataset
      // we try to only choose vectors not equal to zero vector
      tmp = random.nextInt(data.rows.toInt)
      //println("sum = " + data.take(tmp)(0).toArray.sum)
      if(getMatVect(data, tmp).toArray.sum != 0 || nbIt > 5 ) {
        randomIndices += tmp
        nbIt = 0
      } else {
        nbIt += 1
      }

    }

    var centroids: Array[Vect[Double]] = new Array[Vect[Double]](randomIndices.size)
    for(i <- 0 until randomIndices.size)
      centroids(i) = getMatVect(data, randomIndices(i))

    //var clusters = new Array[Array[Vect[Double]]](this.k)(data.cols)

    centroids
    /*

    val myvect = data.zipWithIndex.filter({case (_, index) => randomIndices.contains(index.toInt)}).map({ case (vect, _) => (vect, tmpRDD)}).map(s => s._1).take(1)

    // now we find these vectors in the dataset using their ID, previously randomly selected
    // we add another element to these vectors : which will be the vector of the cluster
    // thus, each cluster is an RDD / map containing a pair (Vector : centroid, Vector: cluster values)
    val myclusters = data
      .zipWithIndex
      .filter({ case (_, index) => randomIndices.contains(index.toInt) })
      .map({ case (centroid, index) => (index.toInt, (centroid, tmpRDD)) }).collectAsMap()
*/
  }

  def calcDistances(centroids: Array[Vect[Double]], vector: Vect[Double]): Array[Double] ={
    centroids.map{ case(c) => vector.distanceTo(c)}
  }

  def getPartialDistances(ids: CM.ArraySeq[Int], centroids: Array[Vect[Double]], data:Array[Vect[Double]]) = {
    data.map{
      case(vect) => (vect, calcDistances(centroids, vect))
    }
  }

  def buildClusters(ids: CM.ArraySeq[Int], centroids: Array[Vect[Double]], data: BMatrix[Double]) = {
    println(ids)
    //val data = newel.filter{ case(id, _) => id == ids}.map{ case(id, mat) => mat}.take(1)

    var newdata = Tensor.blockTensorAsVectors(data)
    //println("count data = " + newdata.count())
    newdata.map({ vect =>
      val byDistanceToCentroid = new Ordering[Vect[Double]] {
        def compare(v1: Vect[Double], v2: Vect[Double]) = v1.distanceTo(vect) compareTo v2.distanceTo(vect)
      }
      (vect, centroids.map({ case(centroid) => centroid }) min byDistanceToCentroid)
    })
      .groupBy({ case(centroid, _) => centroid }).map({ case(centroid, vectorsToCentroids) =>
        val vects = vectorsToCentroids.map({ case(vect, _) => vect })
        (centroid, vects)
      })
  }
  /*
  // returns Map[Vector, RDD[Vector]]
  //@tailrec
  def buildClusters(data: RDD[Vector], prevClusters: scala.collection.Map[Int, (Vector, RDD[Vector])]): scala.collection.Map[Vector, Iterable[Vector]] = {

    println("------------------- Build clusters -------------------- ")
    // get rid of I
    /*val nextClusters = data.map({ vect =>
      val byDistanceToPoint = new Ordering[Vector] {
        //override def compare(v1: Vector, v2: Vector) = distanceTo(v1, vect) compareTo distanceTo(v2, vect)
        @transient def compare(v1: Vector, v2: Vector) = v1.distanceTo(vect) compareTo v2.distanceTo(vect)
      }
      (vect, prevClusters.map({ case (_,(centroid, _)) => centroid }) min byDistanceToPoint)
    }).*/groupBy({ case (centroid, _) => centroid }).map({ case (centroid, pointsToCentroids) =>
      val vects = pointsToCentroids.map({ case (vect, _) => vect })
      (centroid, vects)
    })

    println("---------- end clustering ------ ")

/*
    if (prevClusters != nextClusters) {
      val nextClustersWithBetterCentroids = nextClusters.map({
        case (centroid, members) =>
          val (sum, count) = members.foldLeft((Vector(0, 0, 0), (0))({ case ((acc, c), curr) => (acc sum curr.t, c + 1) })
          (sum divideBy count, members)
      })

      buildClusters(data, nextClustersWithBetterCentroids)
    } else {*/
      //prevClusters
    nextClusters.collectAsMap()
   // }
  }
*/
}