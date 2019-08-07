package project.core.officialApprox

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.storage.StorageLevel._

/**
  * Created by root on 2016/2/1.
  */
object RunTucker {

  def main (args: Array[String]): Unit =
  {
    // Default variable
    var tensorPath = ""
    var approxPath = ""
    var deCompSeq = Array( 0, 0, 0 )
    var coreRank = Array( 0, 0, 0 )
    var maxIter: Int = 20
    var epsilon = 5e-5
    var reconstFlag = 0
    var reconstPath =  ""
    var rddPartitionSize = ""

    // Read input argument
    for( argCount <- 0 until args.length by 2 )
    {
      args( argCount ) match
      {
        case "--TensorPath"  =>
          tensorPath = args( argCount + 1 )
        case "--ApproxPath" =>
          approxPath = args( argCount + 1 )
        case "--DeCompSeq" =>
          deCompSeq = args( argCount + 1 ).split(",").map(_.toInt)
        case "--CoreRank" =>
          coreRank = args( argCount + 1 ).split(",").map(_.toInt)
        case "--MaxIter" =>
          maxIter = args( argCount + 1 ).toInt
        case "--Epsilon" =>
          epsilon = args( argCount + 1 ).toDouble
        case "--DoReconst" =>
          reconstFlag = args( argCount + 1 ).toInt
        case "--ReconstPath" =>
          reconstPath = args( argCount + 1 )
        case "--RDDPartitionSize"=>
          rddPartitionSize = args(argCount + 1)
      }
    }

    //******************************************************************************************************************
    // Section to do tensor Tucker approximation
    //******************************************************************************************************************

    // Set Spark context
    val conf = new SparkConf()
      .setAppName("TensorTucker").setMaster("local[*]")
    val sc = new SparkContext( conf )
    TensorTucker.setSparkContext( sc )
    Tensor.setSparkContext( sc )

    // Read tensor header
    val tensorInfo = Tensor.readTensorHeader( tensorPath )
    // Read tensor block and transform into RDD
    val tensorRDD = Tensor.readTensorBlock( tensorPath, tensorInfo.blockRank , rddPartitionSize)
    tensorRDD.persist( MEMORY_AND_DISK )

    // Run tensor Tucker approximation
    val( coreRDD, coreInfo, bcBasisMatrixArray, iterRecord ) =
      TensorTucker.deComp ( tensorRDD, tensorInfo, deCompSeq, coreRank, maxIter, epsilon )
    println(s"iterRecord, $iterRecord")

    // Save block header and core block
    Tensor.saveTensorHeader( approxPath, coreInfo )
    Tensor.saveTensorBlock( approxPath, coreRDD )
    // Save basis matrices
    Tensor.saveBasisMatrices( approxPath + "Basis/", bcBasisMatrixArray, deCompSeq )

    //******************************************************************************************************************
    // Section to do tensor Tucker reconstruction
    //******************************************************************************************************************

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

  }

}