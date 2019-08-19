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

/**.
  * Creating MySpark object to set only one SparkContext
  * */
object MySpark{

  // Set Spark context
  val conf = new SparkConf()
    .setAppName("TensorTucker").setMaster("local[*]")

  val sc = new SparkContext( conf )
  TensorTucker.setSparkContext( sc )
  //Tensor.setSparkContext( sc )


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
  var clusters: RDD[(CM.ArraySeq[Int], DenseMatrix[Double])] = MySpark.sc.parallelize(Seq(null))
  var clustersInfo: Tensor.TensorInfo = null
  var coreTensors: RDD[(CM.ArraySeq[Int], DenseMatrix[Double])] = MySpark.sc.parallelize(Seq(null))
  var coreInfo: Tensor.TensorInfo = null
  var basisMatrices: Array[Broadcast[DenseMatrix[Double]]] = null
  val deCompSeq2: Array[Int] = Array(1, 2, 3)



  //******************************************************************************************************************
  // Section to do data pre-processing
  //******************************************************************************************************************
  if(clustersPath == "") {
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
    var finalClusters: RDD[Array[(CM.ArraySeq[Int], DenseMatrix[Double])]] = null
    val (finalClusters2, clustersInfo2) = TensorTucker.transformClusters(tensorRDD, MySpark.sc.parallelize(clusteredRDDmat2.map { case (x, y) => y }), clusterMembers, tensorInfo, coreRank.clone(), cluNum, 0)

    finalClusters = finalClusters2
    clustersInfo = clustersInfo2(2)
    var test = finalClusters.zipWithIndex.filter { case (x, y) => y == 2 }.map { case (x, y) => x }.collect()
    clusters = MySpark.sc.parallelize(test(0))

    // save clusters
    Tensor.saveTensorHeader("data/clusters/", clustersInfo)
    Tensor.saveTensorBlock("data/clusters/", clusters)

  } else {
    val clustersInfo2 = Tensor.readTensorHeader(clustersPath)
    clustersInfo = new Tensor.TensorInfo(4, Array(24,71,31,31), Array(50, 50, 14, 14), Array(1, 2, 3, 3), Array(24,21,3,3))

    clusters = Tensor.readTensorBlock(clustersPath, clustersInfo2.blockRank).map{ x => (x._1, new DenseMatrix[Double](x._2.numRows, x._2.numCols, x._2.values))}
  }

  if(basisPath == "" || corePath == ""){
    clustersInfo = new Tensor.TensorInfo(4, Array(24,71,31,31), Array(50, 50, 14, 14), Array(1, 2, 3, 3), Array(24,21,3,3))

    val (coreTensors2, coreInfo2, basisMatrices2, iterRecord) = TensorTucker.deComp2(clusters, clustersInfo, deCompSeq, coreRank, maxIter, epsilon)
    coreTensors = coreTensors2
    coreInfo = coreInfo2
    basisMatrices = basisMatrices2
    // save basis matrices
    //val basisInfo: Tensor.BasisInfo = new Tensor.BasisInfo(basisMatrices.map{ x => x.value.rows}, basisMatrices.map{ x => x.value.cols})
    //Tensor.saveBasisHeader("data/basisMatrices/", basisInfo)
    //Tensor.saveBasisMatrices("data/basisMatrices/", basisMatrices.map{ x => MySpark.sc.broadcast(x.value.reshape(1, x.value.rows * x.value.cols))}, deCompSeq2)
    Tensor.saveBasisMatrices("data/basisMatrices/", basisMatrices, deCompSeq)

    // save core tensors
    Tensor.saveTensorHeader("data/coreTensors/", coreInfo)
    Tensor.saveTensorBlock("data/coreTensors/", coreTensors)
  } else {

    // start with previously calculated coreTensor and basisMatrices
    var basisMatrices2 = Tensor.readBasisMatrices(basisPath, deCompSeq).collect()
    basisMatrices = basisMatrices2.map{ x => MySpark.sc.broadcast(x._2.reshape(x._1, x._2.rows / x._1))}.reverse

    //val basisMatrices = Tensor.readBasisMatrices(basisPath, deCompSeq2).map{ x => MySpark.sc.broadcast(new DenseMatrix[Double](x.numRows, x.numCols, x.values))}
    //var basisMatrices2 = Tensor.readBasisMatrices(basisPath, deCompSeq).map{ x => new DenseMatrix[Double](x.numRows, x.numCols, x.values)}.collect()
    coreInfo = Tensor.readTensorHeader(corePath)
    coreTensors = Tensor.readTensorBlock(corePath, coreInfo.blockRank).map{ x => (x._1, new DenseMatrix[Double](x._2.numRows, x._2.numCols, x._2.values))}

  }

  // for each cluster
  for(c <- cluNum to 1 by -1){
    TensorTucker.computeApprox(clusters, clustersInfo, cluNum, coreTensors, coreInfo, basisMatrices, deCompSeq, 0)
  }

  println("CTA done")
}


//******************************************************************************************************************
// Section to do tensor Tucker approximation
//******************************************************************************************************************
//saveTmpBlock()
// Read tensor header
//val tensorInfo = Tensor.readTensorHeader( tensorPath + "header/part-00000" )



// Run tensor Tucker approximation
/*
    val( coreRDD, coreInfo, bcBasisMatrixArray, iterRecord ) =
      TensorTucker.deComp ( tensorRDD, tensorInfo, deCompSeq, coreRank, maxIter, epsilon )

    println(" ------------------------ this is ok ----------------------------------")
    println(s"iterRecord, $iterRecord")

    // Save block header and core block
    Tensor.saveTensorHeader( approxPath, coreInfo )
    Tensor.saveTensorBlock( approxPath, coreRDD )
    // Save basis matrices
    Tensor.saveBasisMatrices( approxPath + "Basis/", bcBasisMatrixArray, deCompSeq )

*/

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
