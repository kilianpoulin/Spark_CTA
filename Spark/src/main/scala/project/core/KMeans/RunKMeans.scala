package project.core.KMeans

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import project.core.TensorPack._

import scala.collection.{mutable => CM}
import project.core.Tucker.Tensor
import project.core.Tucker.Tensor.MyPartitioner
import project.core.Tucker.TensorTucker
import org.apache.spark.mllib.linalg.{DenseMatrix => DMatrix}
import project.core.Tucker.{MySpark => MySpark}
import project.core.CTA.RunCTA.{args, coreRank}

object RunKMeans extends App {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  MySpark.sc.setLogLevel("ERROR")

  // Default variable
  var tensorPath = "data/donkey_spark/"
  var deCompSeq = Array(0, 0, 0)
  var maxIter: Int = 20
  var centroidsInit = "random"
  var centroidsPath = ""
  var cluNum = 1
  var unfoldDim = 0
  var coreRank: Array[Int] = Array(0,0,0)

  // Read input argument
  for (argCount <- 0 until args.length by 2) {
    args(argCount) match {
      case "--TensorPath" =>
        tensorPath = args(argCount + 1)
      case "--DeCompSeq" =>
        deCompSeq = args(argCount + 1).split(",").map(_.toInt)
      case "--MaxIter" =>
        maxIter = args(argCount + 1).toInt
      case "--CentroidsInit" =>
        centroidsInit = args(argCount + 1)
      case "--CentroidsPath" =>
        centroidsPath = args(argCount + 1)
      case "--CluNum" =>
        cluNum = args(argCount + 1).toInt
      case "--UnfoldDim" =>
        unfoldDim = args(argCount + 1).toInt
      case "--CoreRank" =>
        coreRank = args(argCount + 1).split(",").map(_.toInt)
    }
  }

  //******************************************************************************************************************
  // Run KMeans
  //******************************************************************************************************************

  run(tensorPath, deCompSeq, maxIter, centroidsInit, centroidsPath, cluNum, unfoldDim, coreRank)




  //******************************************************************************************************************
  // Sub-functions
  //******************************************************************************************************************

  /** -----------------------------------------------------------------------------------------------------------------
    * Contains all steps necessary to perform K-Means
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def run(tensorPath: String, deCompSeq: Array[Int], maxIter: Int, centroidsInit: String, centroidsPath: String, cluNum: Int, unfoldDim: Int, coreRank: Array[Int]):
  (RDD[Array[(CM.ArraySeq[Int], DenseMatrix[Double])]], Array[Tensor.TensorInfo], RDD[(CM.ArraySeq[Int], DMatrix)], Tensor.TensorInfo) = {

    //******************************************************************************************************************
    // Initialize variables
    //******************************************************************************************************************
    var clusters: RDD[Array[(CM.ArraySeq[Int], DenseMatrix[Double])]] = MySpark.sc.parallelize(Seq(null))
    var clustersInfo: Array[Tensor.TensorInfo] = Array.fill(cluNum)(null)

    //******************************************************************************************************************
    // Section to do data pre-processing
    //******************************************************************************************************************

    val tuple = GetTensor.extractTensor(tensorPath)
    val tensorRDD = tuple._1
    val tensorInfo = tuple._2

    val tensorBlocks = GetTensor.unfoldTensor(tensorRDD, tensorInfo, unfoldDim)

    //******************************************************************************************************************
    // Section to perform K-Means
    //******************************************************************************************************************

    val kmeans = new KMeansClustering(cluNum, centroidsInit, centroidsPath, maxIter, unfoldDim, tensorInfo.tensorDims, tensorInfo)
    val (clusteredRDD, clusterMembers) = kmeans.train(tensorBlocks)


    //************************************************** ****************************************************************
    // Section to perform initial tensor decomposition
    //******************************************************************************************************************

    // convert Densevectors to DenseMatrix
    var clusteredRDDmat = clusteredRDD.map { x => (x.size, x(0).length, x.map { y => y.toArray }) }
      .map { x => new DenseMatrix(x._2, x._1, x._3.reduce((a, b) => a ++ b)) }.map { x => (CM.ArraySeq[Int](0, 0), x.t) }


    // use vector IDs to create clusters
    val tuple2 = TensorTucker.transformClusters(tensorRDD, clusteredRDDmat.map { _._2.rows }.collect(), clusterMembers,
      tensorInfo, coreRank.clone(), cluNum, 0)

    clusters = tuple2._1
    clustersInfo = tuple2._2

    for (c <- 0 to cluNum - 1) {
      val clInfo = clustersInfo(c)

      val cluster = clusters.zipWithIndex.filter { case (x, y) => y == c }.map { case (x, y) => x }.flatMap { x => x }
        .reduceByKey(new MyPartitioner(clInfo.blockNum), (a, b) => a + b)

      // save cluster
      Tensor.saveTensorHeader("data/clusters/" + c + "/", clInfo)
      Tensor.saveTensorBlock("data/clusters/" + c + "/", cluster)
    }

    (clusters, clustersInfo, tensorRDD, tensorInfo)
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Extracts previous clusters from path
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def extract(clustersPath: String, tensorPath: String):
    (RDD[Array[(CM.ArraySeq[Int], DenseMatrix[Double])]], Array[Tensor.TensorInfo], RDD[(CM.ArraySeq[Int], DMatrix)], Tensor.TensorInfo) ={

    //******************************************************************************************************************
    // Initialize variables
    //******************************************************************************************************************
    var clusters: RDD[Array[(CM.ArraySeq[Int], DenseMatrix[Double])]] = MySpark.sc.parallelize(Seq(null))
    var clustersInfo: Array[Tensor.TensorInfo] = Array.fill(cluNum)(null)

    //******************************************************************************************************************
    // Section to do data pre-processing
    //******************************************************************************************************************
    val tuple = GetTensor.extractTensor(tensorPath)
    val tensorRDD = tuple._1
    val tensorInfo = tuple._2

    for (c <- 0 to cluNum - 1) {
      clustersInfo(c) = Tensor.readTensorHeader(clustersPath + c + "/")

      if (clustersInfo(c).blockRank(unfoldDim) < tensorInfo.blockRank(unfoldDim)) {
        clustersInfo(c).blockRank(unfoldDim) = tensorInfo.blockRank(unfoldDim)
      }

      val oneCl = Tensor.readClusterBlock(clustersPath + c + "/", clustersInfo(c).blockRank)
        .map { s => s.map { x => (x._1, new DenseMatrix[Double](x._2.numRows, x._2.numCols, x._2.values)) } }
      if (c == 0)
        clusters = oneCl
      else
        clusters = clusters.union(oneCl)
    }

    (clusters, clustersInfo, tensorRDD, tensorInfo)
  }

}