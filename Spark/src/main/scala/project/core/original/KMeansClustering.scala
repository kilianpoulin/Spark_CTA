package project.core.original

import java.io.File
import java.lang.Math.{pow, sqrt}

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

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
class KMeansClustering (
                         private var k: Int,
                         private var maxIterations: Int,
                         private var tensorDim: Int) extends Serializable {
/*
  //
  // Creating implicit class to add function distanceTo to Vector
  //
  @transient case class VData(@transient v1: Vector) extends Serializable {
    @transient val distanceTo = (v2: Vector) =>  sqrt((v1.toArray zip v2.toArray).map { case (x,y) => pow(x - y, 2) }.sum)
  }


  @transient implicit def distancecalc(@transient v: Vector): VData = VData(v)
*/
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
    var clustersTmp = createRandomCentroids(ids(0), mat(0))

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

  def getMatVect(data: DMatrix, row: Int): Vect[Double] ={
    var tmpArray = new Array[Double](data.numCols)
    for(i <- 0 until data.numCols){
      tmpArray(i) = data.apply(row, i)
    }
    tmpArray.toVector
  }

  // should return scala.collection.Map[Vector, RDD[Vector]]
  // Array[Array[Vector]]
  def createRandomCentroids(ids: CM.ArraySeq[Int], data: DMatrix) = {

    val randomIndices = ListBuffer[Int]()
    val random = new Random()
    var tmp = 0
    while (randomIndices.size < k) {
      // choosing on of the vectors in the dataset to become one cluster centroid
      // so we will choose k vectors in the dataset
      // we only choose vectors not equal to zero vector
      tmp = random.nextInt(data.numRows.toInt)
      //println("sum = " + data.take(tmp)(0).toArray.sum)
      if(getMatVect(data, tmp).toArray.sum != 0)
        randomIndices += tmp
    }

    var partCluster = MySpark.sc.parallelize(Seq(Seq(mutable.ArraySeq[Int](k - 1, 0), getMatVect(data, randomIndices(0))), Seq(mutable.ArraySeq[Int](k, 0), getMatVect(data, randomIndices(1)))))

    //var clusters = new Array[Array[Vect[Double]]](this.k)(data.cols)

    val clusters = partCluster
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
    clusters
  }

/*
  // returns Map[Vector, RDD[Vector]]
  //@tailrec
  def buildClusters(data: RDD[Vector], prevClusters: scala.collection.Map[Int, (Vector, RDD[Vector])]): scala.collection.Map[Vector, Iterable[Vector]] = {

    println("------------------- Build clusters -------------------- ")
    // get rid of I
    val nextClusters = data.map({ vect =>
      val byDistanceToPoint = new Ordering[Vector] {
        //override def compare(v1: Vector, v2: Vector) = distanceTo(v1, vect) compareTo distanceTo(v2, vect)
        @transient def compare(v1: Vector, v2: Vector) = v1.distanceTo(vect) compareTo v2.distanceTo(vect)
      }
      (vect, prevClusters.map({ case (_,(centroid, _)) => centroid }) min byDistanceToPoint)
    }).groupBy({ case (centroid, _) => centroid }).map({ case (centroid, pointsToCentroids) =>
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