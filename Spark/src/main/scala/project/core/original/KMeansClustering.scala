package project.core.original

import java.lang.Math.{pow, sqrt}

import org.apache.spark.rdd.RDD

import scala.math.Ordering
import scala.collection.{mutable => CM}
import org.apache.spark.mllib.linalg.{DenseMatrix => DMatrix}

import scala.util.Random
import scala.{Vector => Vect}
import project.core.original.RunTucker.{tensorInfo, tensorRDD}
import project.core.original.Tensor.{MyPartitioner, localTensorUnfoldBlock}
import breeze.linalg.{DenseMatrix => BMatrix}

import scala.collection.mutable.{ListBuffer}
class KMeansClustering (
                         private var k: Int,
                         private var maxIterations: Int,
                         private var tensorDim: Int) extends Serializable {

  /** -----------------------------------------------------------------------------------------------------------------
    * Implicit class to add function distanceTo to Vector class
    * -----------------------------------------------------------------------------------------------------------------
    * */
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

    centroids
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Returns the distance from one vector to each centroid vector
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def calcDistances(centroids: Array[Vect[Double]], vector: Vect[Double]): Array[Double] ={
    centroids.map{ case(c) => vector.distanceTo(c)}
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Returns a tuple containing a vector and all the distances from this vector to each centroid vector
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def getPartialDistances(ids: CM.ArraySeq[Int], centroids: Array[Vect[Double]], data:Array[Vect[Double]]) = {
    data.map{
      case(vect) => (vect, calcDistances(centroids, vect))
    }
  }


  /** -----------------------------------------------------------------------------------------------------------------
    * Main function for K-Means
    * Contains all steps
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def run(data: RDD[ (CM.ArraySeq[Int], DMatrix) ]){

    // Step 1 : Unfold each block and make one partition for one block
    val newdata = data.map{ case(ids, mat) =>
      (ids, localTensorUnfoldBlock( mat, ids, 0,  tensorInfo.blockRank, tensorInfo.blockNum, tensorInfo.tensorRank))
    }.reduceByKey( new MyPartitioner( tensorInfo.blockNum ), ( a, b ) => a + b )

    println(" (3) [OK] Tensor unfolded along Dimension 1 ")

    // Step 2 : create k clusters with random values as centroids data)
    var centroids = newdata.map{ case(ids, mat) => (ids, createRandomCentroids(mat))}

    println(" (4) [OK] " + k + " splitted clusters successfully initialized ")


    // Step 3 : calculate distance between partial vectors and each partial centroid vector
    var distances = centroids.join(newdata).map { case (ids, (centroids, values)) => (ids, getPartialDistances(ids, centroids, Tensor.blockTensorAsVectors(values))) }
    println(" (5) [OK] calculating distances between partial vectors and each partial centroid vector ")
    // display list of distances
    //distances.map{ case(id, values) => values }.flatMap(v => v).map{ case(vector, distances) => distances}.flatMap(a => a).collect().foreach(println)


    // Step 4 : Addition partial distances
    // adding partial distances
    var tmpDistances = distances.map{ x => (getVectorIds(x._1), x._2) }.groupByKey().map{case (i, iter) => (i, iter.toList)}
    var fullDistances = tmpDistances.map{ case(i, iter) => (i, combinePartialDistances(iter))}

    //test.map{ case(i, iter) => combinePartialVectors(iter).size}.foreach(println)
    println(" (6) [OK] Addition partial distances for each full vector ")

    /* IN PROGRESS
    // Step 5 : Build clusters based on the shortest distance of the full vector
      // we need to break the ids of fullDistances
      var tmpfullDistances = fullDistances.map{case(x) => (x._1, x._2)}.flatMap{ case(a,b) => a.map(y => (y, b))}
      //println("test = " + test.size)
      var tmpclusters = centroids.join(tmpfullDistances).join(distances.map{ case(ids, x) => (ids, x.map{ case(centroids, vectors) => (centroids, vectors)})}).collect
        var clusters = tmpclusters
        .map{ case(ids, o) => (ids, buildClusters(ids, o._1._1, o._2.map{ case(c, d) => c},  o._1._2))}
    // centroids, vectors, distances
      //var clusters = centroids.join(newdata).map{ case(ids, (centroids, values)) => (ids, buildClusters(ids, centroids, values))}

      println(" (7) [OK] Building clusters ")
      */

  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Dispatches partial vectors into each partial cluster
    * -----------------------------------------------------------------------------------------------------------------
    * */
  /*
  def buildClusters(ids: CM.ArraySeq[Int], centroids: Array[Vect[Double]], data: Array[Vect[Double]], distances: Iterable[List[Double]]) = {
    println(ids)
    //val data = newel.filter{ case(id, _) => id == ids}.map{ case(id, mat) => mat}.take(1)

    //var newdata = Tensor.blockTensorAsVectors(data)
    //println("count data = " + newdata.count())
    data.map({ vect =>
      val byDistanceToCentroid = new Ordering[Double] {
        def compare(v1: Double, v2: Double) = v1 compareTo v2
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
* */
}