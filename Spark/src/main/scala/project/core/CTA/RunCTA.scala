package project.core.CTA

import project.core.TensorPack._
import project.core.KMeans._
import project.core.Tucker.{Tensor, TensorTucker}
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{DenseMatrix => DMatrix}

import scala.collection.{mutable => CM}
import project.core.Tucker.Tensor.MyPartitioner
import project.core.Tucker.{MySpark => MySpark}
/**.
  * Creating MySpark object to set only one SparkContext
  * */
object RunCTA extends App {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  MySpark.sc.setLogLevel("ERROR")

  // Default variable
  var tensorPath = "data/donkey_spark/"
  var approxPath = ""
  var deCompSeq = Array(0, 0, 0)
  var coreRank = Array(0, 0, 0)
  var maxIter: Int = 20
  var epsilon = 5e-5
  var reconstFlag = 0
  var reconstPath = ""
  var centroidsInit = "random"
  var centroidsPath = ""
  var cluNum = 1
  var basisPath = ""
  var corePath = ""
  var clustersPath = ""
  var finalPath = ""
  var unfoldDim = 0
  val saveBasisPath = "data/basisMatrices/"
  val saveCorePath = "data/coreTensors/"

  // Read input argument
  for (argCount <- 0 until args.length by 2) {
    args(argCount) match {
      case "--TensorPath" =>
        tensorPath = args(argCount + 1)
      case "--ApproxPath" =>
        approxPath = args(argCount + 1)
      case "--DeCompSeq" =>
        deCompSeq = args(argCount + 1).split(",").map(_.toInt)
      case "--CoreRank" =>
        coreRank = args(argCount + 1).split(",").map(_.toInt)
      case "--MaxIter" =>
        maxIter = args(argCount + 1).toInt
      case "--Epsilon" =>
        epsilon = args(argCount + 1).toDouble
      case "--DoReconst" =>
        reconstFlag = args(argCount + 1).toInt
      case "--ReconstPath" =>
        reconstPath = args(argCount + 1)
      case "--CentroidsInit" =>
        centroidsInit = args(argCount + 1)
      case "--CentroidsPath" =>
        centroidsPath = args(argCount + 1)
      case "--CluNum" =>
        cluNum = args(argCount + 1).toInt
      case "--CorePath" =>
        corePath = args(argCount + 1)
      case "--BasisPath" =>
        basisPath = args(argCount + 1)
      case "--ClustersPath" =>
        clustersPath = args(argCount + 1)
      case "--FinalPath" =>
        finalPath = args(argCount + 1)
      case "--UnfoldDim" =>
        unfoldDim = args(argCount + 1).toInt
    }
  }

//******************************************************************************************************************
// Initialize variables
//******************************************************************************************************************
  var tensorRDD: RDD[(CM.ArraySeq[Int], DMatrix)] = MySpark.sc.parallelize(Seq(null))
  var tensorInfo: Tensor.TensorInfo = null
  var clusters: RDD[Array[(CM.ArraySeq[Int], DenseMatrix[Double])]] = MySpark.sc.parallelize(Seq(null))
  var clustersInfo: Array[Tensor.TensorInfo] = Array.fill(cluNum)(null)
  var coreTensors: Array[RDD[(CM.ArraySeq[Int], DenseMatrix[Double])]] = Array.fill(cluNum)(MySpark.sc.parallelize(Seq(null)))
  var coreInfo: Array[Tensor.TensorInfo] = Array.fill(cluNum)(null)
  var basisMatrices: Array[Array[Broadcast[DenseMatrix[Double]]]] = Array.fill(cluNum)(null)
  val deCompSeq2: Array[Int] = Array(1, 2, 3)
  var reconstTensor: RDD[(CM.ArraySeq[Int], DenseMatrix[Double])] = MySpark.sc.parallelize(Seq(null))


//******************************************************************************************************************
// Section to perform K-Means
//******************************************************************************************************************

  if(clustersPath == ""){

    val tuple = RunKMeans.run( tensorPath, deCompSeq, maxIter, centroidsInit, centroidsPath, cluNum, unfoldDim, coreRank)
    clusters = tuple._1
    clustersInfo = tuple._2
    tensorRDD = tuple._3
    tensorInfo = tuple._4

  } else {

    RunKMeans.cluNum = cluNum
    val tuple = RunKMeans.extract(clustersPath, tensorPath)
    clusters = tuple._1
    clustersInfo = tuple._2
    tensorRDD = tuple._3
    tensorInfo = tuple._4

  }


//******************************************************************************************************************
// Section to perform CTA
//******************************************************************************************************************

  val approxErr: DenseMatrix[Double] = DenseMatrix.zeros[Double](tensorInfo.tensorRank(unfoldDim), cluNum)

  if(corePath == "" && basisPath == ""){

    // decomposition for each cluster
    deCompClusters()

  } else {
    extract(corePath, basisPath)
  }

  // run approximation for N iterations or until convergence
  NiterApprox()


  println("")


  /** -----------------------------------------------------------------------------------------------------------------
    * Extracts previous basis matrices and core tensorss
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def extract(corePath: String, basisPath: String): Unit ={

    for (c <- 0 to cluNum - 1) {

      basisMatrices(c) = Tensor.readBasisMatrices(basisPath + c + "/", deCompSeq).map { x => MySpark.sc.broadcast(x) }
      coreInfo(c) = Tensor.readTensorHeader(corePath + c + "/")
      coreTensors(c) = Tensor.readTensorBlock(corePath + c + "/", coreInfo(c).blockRank).map { x => (x._1, new DenseMatrix[Double](x._2.numRows, x._2.numCols, x._2.values)) }
    }
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Performs decomposition of each cluster based on K-Means clustering
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def deCompClusters() ={

    for (c <- 0 to cluNum - 1) {

      val tuple = TensorTucker.deComp(clusters.zipWithIndex.filter { _._2 == c }.map { _._1 }.flatMap { x => x }, clustersInfo(c), deCompSeq, coreRank, maxIter, epsilon)
      coreTensors(c) = tuple._1
      coreInfo(c) = tuple._2
      basisMatrices(c) = tuple._3

      // save basis matrices
      Tensor.saveBasisMatrices(saveBasisPath + c + "/", basisMatrices(c), deCompSeq)

      // save core tensors
      Tensor.saveTensorHeader(saveCorePath + c + "/", coreInfo(c))
      Tensor.saveTensorBlock(saveCorePath + c + "/", coreTensors(c))
    }

  }

  def NiterApprox(): Unit ={

      val initTensor = tensorRDD.map { case (x, y) => (x, new DenseMatrix[Double](y.numRows, y.numCols, y.values)) }

      for (iter <- 0 to maxIter) {

        for (c <- 0 to cluNum - 1) {
          approxErr(::, c) := TensorTucker.computeApprox(initTensor, tensorInfo, cluNum, coreTensors(c), coreInfo(c), basisMatrices(c), deCompSeq, 0)
        }

        // find cluster with best approximation for each slice
        var cluIds: Array[Int] = Array.fill(tensorInfo.tensorRank(unfoldDim))(0)
        val TapproxErr = approxErr.t
        for (i <- 0 to tensorInfo.tensorRank(unfoldDim) - 1) {
          cluIds(i) = TapproxErr(::, i).argmax
        }

        val cluRanks = cluIds.groupBy(x => x).map { x => x._2.size }.toArray

        // get new clusters
        // use vector IDs to create clusters
        val cluTuple = TensorTucker.transformClusters(tensorRDD, cluRanks, cluIds,
          tensorInfo, coreRank.clone(), cluNum, 0)

        clusters = cluTuple._1
        clustersInfo = cluTuple._2

        for (c <- 0 to cluNum - 1) {
          val clInfo = clustersInfo(c)


          if (clustersInfo(c).blockRank(unfoldDim) < tensorInfo.blockRank(unfoldDim)) {
            clustersInfo(c).blockRank(unfoldDim) = tensorInfo.blockRank(unfoldDim)
          }
          val cluster = clusters.zipWithIndex.filter { case (x, y) => y == c }.map { case (x, y) => x }.flatMap { x => x }
            .reduceByKey(new MyPartitioner(clInfo.blockNum), (a, b) => a + b)

          val tuple = TensorTucker.deComp(clusters.zipWithIndex.filter { case (x, y) => y == c }.map { case (x, y) => x }.flatMap { x => x }, clustersInfo(c), deCompSeq, coreRank, maxIter, epsilon)
          coreTensors(c) = tuple._1
          coreInfo(c) = tuple._2
          basisMatrices(c) = tuple._3

        }

      }


      for (c <- 0 to cluNum - 1) {
        // save basis matrices
        Tensor.saveBasisMatrices("data/basisMatrices/final/" + c + "/", basisMatrices(c), deCompSeq)

        // save core tensors
        Tensor.saveTensorHeader("data/coreTensors/final/" + c + "/", coreInfo(c))
        Tensor.saveTensorBlock("data/coreTensors/final/" + c + "/", coreTensors(c))
      }

  }
}



