package project.core.v2

import java.lang.Math.{pow, sqrt}

import org.apache.spark.rdd.RDD

import scala.math.Ordering
import scala.collection.{mutable => CM}
import org.apache.spark.mllib.linalg.{DenseMatrix => DMatrix}

import scala.util.Random
import scala.{Vector => Vect}
import project.core.v2.RunTucker.{tensorInfo, tensorRDD}
import breeze.linalg.{DenseVector, sum, DenseMatrix => BMatrix}
class KMeansClustering (
                         private var k: Int,
                         private var maxIterations: Int,
                         private var centroidsInit: String,
                         private var centroidsPath: String,
                         private var maxIter: Int,
                         private var tensorDim: Int) extends Serializable {

  /** -----------------------------------------------------------------------------------------------------------------
    * Implicit class to add function distanceTo to Vector class
    * -----------------------------------------------------------------------------------------------------------------
    * */
  @transient case class VData(@transient v1: Vect[Double]) extends Serializable {
    @transient val distanceTo = (v2: Vect[Double]) => {

      val s: Array[Double] = v1.toArray
      val t: Array[Double] = v2.toArray
      var tmp = (s zip t).map{ case (x,y) => x - y }
      var res = tmp.map{ case(x) => pow(x, 2)}
      var resum = res.sum

      resum
    }
  }

  @transient implicit def distancecalc(@transient v: Vect[Double]): VData = VData(v)

  def sum(list: List[Double]): Double = list match {
    case Nil => 0
    case x :: xs => x + sum(xs)
  }

  def train(
             data: RDD[ (CM.ArraySeq[Int], BMatrix[Double]) ]
           ) = {
    run(data)
  }

  /**
    * Set the number of clusters to create (k).
    *
    */
  def setK(k: Int): this.type = {
    require(k > 0,
      s"Number of clusters must be positive but got ${k}")
    this.k = k
    this
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Main function for K-Means
    * Contains all steps
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def getTupleList(list: List[(List[Double], Int)])={
    var newlist = list
    newlist.toArray.map{ case(l, id) => list.filter{ case(l1, id1) => id1 == id}.map(_._1)}.distinct.map{ case(x) => x.transpose.map(_.sum)}.flatMap(x => x).toList
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Combines and addition partial distances to get the distance to each centroid for a full vector
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def combinePartialDistances(list: List[Array[(Vect[Double], Array[Double])]]) ={
    list.toArray.map{ case(x) => x.map{ case (y, z) => z.toList}.zipWithIndex.toList}.toList.flatMap(s => s).groupBy(e => e._2).map(_._2).map{ case(x) => getTupleList(x)}
    //map{case(a) => a.toArray.map(_._1).flatten}.transpose.map(_.sum)
    //.map(_._1)
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Vectors are cut into partial vectors, each in different blocks
    * This function finds all blocks by their ID, that contain a partial vector of a vector.
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def getVectorIds(ids: CM.ArraySeq[Int]): List[CM.ArraySeq[Int]] ={
    var listIds = new Array[CM.ArraySeq[Int]](tensorInfo.blockNum(1))
    for(i <- 0 until tensorInfo.blockNum(1)){
      listIds(i) = ids.clone()
      listIds(i)(1) = i
    }
    listIds.toList
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Returns a vector in a matrix, based on its row number
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def getMatVect(data: BMatrix[Double], row: Int): Vect[Double] ={
    var tmpArray = new Array[Double](data.cols)
    for(i <- 0 until data.cols){
      tmpArray(i) = data.apply(row, i)
    }
    tmpArray.toVector
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Initializes partial centroids for each cluster
    * Each partial centroid is in fact a randomly chosen vector in the corresponding block
    * nbIt is the number of iterations we want to do in order to try to find a vector that different from the vector 0
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def createRandomCentroids(data: RDD[(CM.ArraySeq[Int], BMatrix[Double])], option: String = "") = {
    var centroids: Array[Array[Vect[Double]]] = null

    centroidsInit match {
      case "sample" =>
        var blockValueSize = data.filter { case (x, y) => x == CM.ArraySeq[Int](0, 0) }.map { case (x, y) => y.cols }.collect()
        //var blockIds = data.map{ case(ids, values) => ids(1)}.collect()
        centroids = createCentroidsFromSample(centroidsPath, blockValueSize(0))

      /*case "random" =>
        centroids = data.map{ case(ids, mat) => (ids, createRandomCentroids(mat, "default"))}.collect()*/
    }
    /*
    } else {
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
        if (getMatVect(data, tmp).toArray.sum != 0 || nbIt > 5) {
          randomIndices += tmp
          nbIt = 0
        } else {
          nbIt += 1
        }
      }

      var centroids: Array[Vect[Double]] = new Array[Vect[Double]](randomIndices.size)
      for (i <- 0 until randomIndices.size)
        centroids(i) = getMatVect(data, randomIndices(i))
    }*/
    centroids
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Returns the distance from one vector to each centroid vector (output : k distances)
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def calcDistances(ids: CM.ArraySeq[Int], centroids: Array[Vect[Double]], vector: Vect[Double]): Array[Double] ={
    //centroids.map{ case(c) => vector.distanceTo(c((ids(1))))}
    centroids.map{ case(c) => vector.distanceTo(c)}
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Returns all the distances from a vector to each centroid vector (output : k distances)
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def getPartialDistances(ids: CM.ArraySeq[Int], centroids: Array[Vect[Double]], data:Array[Vect[Double]]) = {
    data.map{
      case(vect) => calcDistances(ids, centroids, vect)
    }
  }

  def combineDists(x: Array[Array[Double]]): Unit ={
    x.map{ case(i) => i}
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Used for creating initial centroids with sample data
    * Retrieves data from files and splits each centroid into blocks
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def createCentroidsFromSample(path: String, blockValueSize: Int) ={
    var cent1 = MySpark.sc.textFile("../Spark/data/clustertmp/cluster1full").collect()
    var cent2 = MySpark.sc.textFile("../Spark/data/clustertmp/cluster2full").collect()
    var cent3 = MySpark.sc.textFile("../Spark/data/clustertmp/cluster3full").collect()
    var centRDD = MySpark.sc.parallelize(Array(cent1, cent2, cent3))

    // Creating blocks of same size as unfolded tensor
    var centroids = centRDD.map{ case(cent) => cent.map{ case(x) => x.split(",")}.flatMap(x => x).map{ case(y) => y.toDouble}.toList.grouped(blockValueSize).map{case (z) => z.toVector}.toArray}.collect()

    centroids
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Sums partial distances previously calculated in each tensor block
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def getFullDistances(array: Array[Array[Array[Double]]]): Array[Array[Double]] ={
    array.map{ case(x) => x.transpose}.map{ case(x) => x.map{ y => y.sum}}
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Main function for K-Means
    * Contains all steps
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def run(data: RDD[ (CM.ArraySeq[Int], BMatrix[Double]) ]) {


    // Step 1 : create k clusters with random or default values as centroids data)
    var centroids: Array[Array[Vect[Double]]] = createRandomCentroids(data, centroidsInit)

    println(" (4) [OK] " + k + " block cluster centroids successfully initialized ")

    println(" (X) [Ready for " + maxIter + " iterations")

    for (iter <- 0 until maxIter) {

      println("\n ... Starting iteration #" + (iter + 1) + " ...\n")

      // Step 2 : Calculate partial distances between block vectors and block centroids
      var partdist = data.map { case (ids, values) => (ids, getPartialDistances(ids, centroids.map { case (x) => x.zipWithIndex.filter { case (cvalues, cids) => cids == ids(1) }.map { case (cvalues, cids) => cvalues } }.flatMap { x => x }, Tensor.blockTensorAsVectors(values))) }.collect()

      println(" (5) [OK] Calculating partial distances between block vectors and block centroid vectors ")

      // Step 3 : Add partial distances and get full vector distances
      var distances2x1 = partdist.groupBy { case (ids, values) => ids(0) }.map { x => x._2.map(y => y._2) }.toArray
      var fulldist = distances2x1.map { x => getFullDistances(x.transpose) }

      println(" (6) [OK] Calculating full distances between tensor vectors and centroids ")

      // Step 4 : Distributing cluster IDs to tensor vectors based on shortest distance to centroids
      var clusters = fulldist.reverse.flatMap { x => x }.map { case (x) => x.indexOf(x.min) }
      println(" (7) [OK] Tensor vectors distributed to closest cluster ")

      // Step 5 : Updating cluster centroids
      for (i <- 0 until k) {
        var members = clusters.zipWithIndex.filter { case (x, y) => x == i }.map { case (x, y) => y }
        var membersv2 = members.filter { x => x > tensorInfo.blockRank(0) - 1 }.map { x => x % tensorInfo.blockRank(0) }

        // get vectors with the corresponding member ids  (transform matrix into vector) for each part (x 34116 and x 34115)
        var tmpvectorsdim1 = data.filter { case (ids, values) => ids(0) == 0 }
          .map { case (ids, values) => Tensor.blockTensorAsVectors(values).zipWithIndex }.map { case (x) => x.filter { case (a, b) => members.contains(b) }.map { case (a, b) => a } }.collect()

        var tmpvectorsdim2 = data.filter { case (ids, values) => ids(0) == 1 }
          .map { case (ids, values) => Tensor.blockTensorAsVectors(values).zipWithIndex }.map { case (x) => x.filter { case (a, b) => membersv2.contains(b) }.map { case (a, b) => a } }.collect()

        var fullvectorsdim = (tmpvectorsdim1 zip tmpvectorsdim2).map { case (x) => x._1 ++ x._2 }

        // Sum each point and divide the result by the number of cluster members => Means
        var bc = fullvectorsdim.toVector.map { case (x) => x.toVector.transpose.map(_.sum / members.size) }
        centroids(i) = bc.toArray
        println(" (8." + (i + 1) + ") [OK] Cluster centroid #" + k + " updated successfully")
      }

      println(" (8+) All cluster centroids successfully updated")
    }

    return
  }
}