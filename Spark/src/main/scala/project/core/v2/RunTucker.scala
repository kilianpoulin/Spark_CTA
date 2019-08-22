package project.core.v2

import java.util

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import project.core.original.RunTucker.{tensorInfo, tensorRDD, unfoldDim}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.collection.{mutable => CM}
import project.core.original.Tensor
import project.core.v2.Tensor.MyPartitioner

/**.
  * Creating MySpark object to set only one SparkContext
  * */
object MySpark{

  // Set Spark context
  val conf = new SparkConf()
    .setAppName("TensorTucker").setMaster("local[*]")

  val sc = new SparkContext( conf )
  TensorTucker.setSparkContext( sc )
  Tensor.setSparkContext( sc )


  val Spark = SparkSession
    .builder()
    .appName("CTA Algorithm")
    .getOrCreate()
}
object RunTucker extends App {

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
     }
  }

  //******************************************************************************************************************
  // Initialize variables
  //******************************************************************************************************************
  var clusters: RDD[Array[(CM.ArraySeq[Int], DenseMatrix[Double])]] = MySpark.sc.parallelize(Seq(null))
  var clustersInfo: Array[Tensor.TensorInfo] = Array.fill(cluNum)(null)
  var coreTensors: Array[RDD[(CM.ArraySeq[Int], DenseMatrix[Double])]] = Array.fill(cluNum)(MySpark.sc.parallelize(Seq(null)))
  var coreInfo: Array[Tensor.TensorInfo] = Array.fill(cluNum)(null)
  var basisMatrices: Array[Array[Broadcast[DenseMatrix[Double]]]] = Array.fill(cluNum)(null)
  val deCompSeq2: Array[Int] = Array(1, 2, 3)



  //******************************************************************************************************************
  // Section to do data pre-processing
  //******************************************************************************************************************

    val tensorInfo = Tensor.readTensorHeader(tensorPath)
    println(" (1) [OK] Read Tensor header ")

    // Read tensor block and transform into RDD
    val tensorRDD = Tensor.readTensorBlock(tensorPath, tensorInfo.blockRank)
    println(" (2) [OK] Read Tensor block ")

    val dmat = tensorRDD.map { case (x, y) => y }.take(1)

    // Unfold the tensor -- pre-process before applying K-Means
    val tensorBlocks = Tensor.TensorUnfoldBlocks2(tensorRDD, unfoldDim, tensorInfo.tensorRank, tensorInfo.blockRank, tensorInfo.blockNum)
    println(" (3) [OK] Block-wise tensor unfolded along dimension " + unfoldDim)

    tensorRDD.unpersist()
    tensorBlocks.persist(MEMORY_AND_DISK)
  if(clustersPath == "") {
    //******************************************************************************************************************
    // Section to perform K-Means
    //******************************************************************************************************************

    val kmeans = new KMeansClustering(cluNum, centroidsInit, centroidsPath, maxIter, tensorInfo.tensorDims, tensorInfo)
    val (clusteredRDD, clusterMembers) = kmeans.train(tensorBlocks)


    //************************************************** ****************************************************************
    // Section to perform CTA
    //******************************************************************************************************************

    // convert Densevectors to DenseMatrix
    var clusteredRDDmat = clusteredRDD.map { x => (x.size, x(0).length, x.map { y => y.toArray }) }.collect()
    var clusteredRDDmat2 = clusteredRDDmat.map { x => new DenseMatrix(x._2, x._1, x._3.reduce((a, b) => a ++ b)) }.map { x => (CM.ArraySeq[Int](0, 0), x.t) }


    // use vector IDs to create clusters
    val tuple = TensorTucker.transformClusters(tensorRDD, clusteredRDDmat2.map{ case (x, y) => y.rows }, clusterMembers,
      tensorInfo, coreRank.clone(), cluNum, 0)

    clusters = tuple._1
    clustersInfo = tuple._2

    for(c <- 0 to cluNum - 1){
      val clInfo = clustersInfo(c)

      val cluster = clusters.zipWithIndex.filter { case (x, y) => y == c }.map { case (x, y) => x }.flatMap{ x => x}
        .reduceByKey( new MyPartitioner( clInfo.blockNum ), ( a, b ) => a + b )

      // save cluster
      Tensor.saveTensorHeader("data/clusters/" + c + "/", clInfo)
      Tensor.saveTensorBlock("data/clusters/" + c + "/", cluster)
    }

  } else {

    for (c <- 0 to cluNum - 1) {
      clustersInfo(c) = Tensor.readTensorHeader(clustersPath + c + "/")

      if (clustersInfo(c).blockRank(unfoldDim) < tensorInfo.blockRank(unfoldDim)) {
        clustersInfo(c).blockRank(unfoldDim) = tensorInfo.blockRank(unfoldDim)
      }

      val oneCl = Tensor.readClusterBlock(clustersPath + c + "/", clustersInfo(c).blockRank)
        .map{ s => s.map{ x => (x._1, new DenseMatrix[Double](x._2.numRows, x._2.numCols, x._2.values)) }}
      if(c == 0)
        clusters = oneCl
      else
      clusters = clusters.union(oneCl)
    }
  }

  if(basisPath == "" || corePath == "") {

    for (c <- 0 to cluNum - 1) {
      val tuple = TensorTucker.deComp2(clusters.zipWithIndex.filter{ case(x,y) => y == c}.map{ case(x,y) => x}.flatMap{x => x}, clustersInfo(c), deCompSeq, coreRank, maxIter, epsilon)
      coreTensors(c) = tuple._1
      coreInfo(c) = tuple._2
      basisMatrices(c) = tuple._3

      // save basis matrices
      Tensor.saveBasisMatrices("data/basisMatrices/" + c + "/", basisMatrices(c), deCompSeq)

      // save core tensors
      Tensor.saveTensorHeader("data/coreTensors/" + c + "/", coreInfo(c))
      Tensor.saveTensorBlock("data/coreTensors/" + c + "/", coreTensors(c))
    }

  } else {

    for(c <- 0 to cluNum - 1){

      // start with previously calculated coreTensor and basisMatrices
      basisMatrices(c) = Tensor.readBasisMatrices(basisPath + c + "/", deCompSeq).map{ x => MySpark.sc.broadcast(x)}
      coreInfo(c) = Tensor.readTensorHeader(corePath + c + "/")
      coreTensors(c) = Tensor.readTensorBlock(corePath + c + "/", coreInfo(c).blockRank).map{ x => (x._1, new DenseMatrix[Double](x._2.numRows, x._2.numCols, x._2.values))}
    }

  }

  //******************************************************************************************************************
  // Section to perform tensor approximation
  //******************************************************************************************************************

  val initTensor = tensorRDD.map{ case(x,y) => (x, new DenseMatrix[Double](y.numRows, y.numCols, y.values))}

  val approxErr: DenseMatrix[Double] = DenseMatrix.zeros[Double](tensorInfo.tensorRank(unfoldDim), cluNum)

  for(iter <- 0 to maxIter){
    // for each cluster
    for(c <- 0 to cluNum - 1){
      approxErr(::, c) := TensorTucker.computeApprox(initTensor, tensorInfo, cluNum, coreTensors(c), coreInfo(c), basisMatrices(c), deCompSeq, 0)
    }

    // find max
    var cluIds: Array[Int] = Array.fill(tensorInfo.tensorRank(unfoldDim))(0)
    val TapproxErr = approxErr.t
    for(i <- 0 to tensorInfo.tensorRank(unfoldDim) - 1){
      cluIds(i) = TapproxErr(::, i).argmax
    }

    val cluRanks = cluIds.groupBy(x => x).map{ x => x._2.size}.toArray
   // val cluRanks = cluIds.groupBy(x => x).map{ x => (x._1, x._2.size)}.toArray.sortBy(x => x._1).map{ x => x._2}
   // val finalCluIDs = cluIds.map{ x => x}
      // get new clusters
      // use vector IDs to create clusters
      val cluTuple = TensorTucker.transformClusters(tensorRDD, cluRanks, cluIds,
        tensorInfo, coreRank.clone(), cluNum, 0)

      clusters = cluTuple._1
      clustersInfo = cluTuple._2

      for(c <- 0 to cluNum - 1){
        val clInfo = clustersInfo(c)

        val cluster = clusters.zipWithIndex.filter { case (x, y) => y == c }.map { case (x, y) => x }.flatMap{ x => x}
          .reduceByKey( new MyPartitioner( clInfo.blockNum ), ( a, b ) => a + b )

        // save cluster
       // Tensor.saveTensorHeader("data/clusters/" + c + "/", clInfo)
        //Tensor.saveTensorBlock("data/clusters/" + c + "/", cluster)

        val tuple = TensorTucker.deComp2(clusters.zipWithIndex.filter{ case(x,y) => y == c}.map{ case(x,y) => x}.flatMap{x => x}, clustersInfo(c), deCompSeq, coreRank, maxIter, epsilon)
        coreTensors(c) = tuple._1
        coreInfo(c) = tuple._2
        basisMatrices(c) = tuple._3
      }

    }

}


//******************************************************************************************************************
// Section to do tensor Tucker reconstruction
//******************************************************************************************************************
/*
  // Reconstruct input tensor
  if( reconstFlag == 1 )
  {
    val( reconstRDD, reconstInfo ) = TensorTucker.reConst ( coreRDD, coreInfo, deCompSeq, bcBasisMatrixArray )

    // Save reconst block header and reconst block
    Tensor.saveTensorHeader( reconstPath, reconstInfo )
    Tensor.saveTensorBlock( reconstPath, reconstRDD )
  }

  // Shut down Spark context
  sc.stop
  */
