package project.core.v2

import org.apache.log4j.{Level, Logger}
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import project.core.original.RunTucker.{tensorInfo, tensorRDD, unfoldDim}
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
    }
  }

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


//******************************************************************************************************************
// Section to perform K-Means
//******************************************************************************************************************

  val kmeans = new KMeansClustering(3, 5, centroidsInit, centroidsPath, maxIter, tensorInfo.tensorDims)
  kmeans.train(tensorBlocks)

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
