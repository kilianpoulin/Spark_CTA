package project.core.original

import org.apache.log4j.{Level, Logger}
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.{SparkConf, SparkContext}
import Tensor._
import org.apache.spark.rdd._
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vectors, DenseMatrix => DMatrix, Vector => MVector}
import org.apache.spark.sql.SparkSession
import project.core.original.KMeansClustering

/**
  * Created by root on 2016/2/1.
  */
object RunTucker extends App{


  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)


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




  import Spark.implicits._
    // Default variable
    var tensorPath = "data/donkey_spark/"
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
    //saveTmpBlock()
    // Read tensor header
    //val tensorInfo = Tensor.readTensorHeader( tensorPath + "header/part-00000" )
    val tensorInfo = Tensor.readTensorHeader("data/ConvertedData/header/part-00000" )
    // Read tensor block and transform into RDD
    println("------------- OK 1 --------------------------------")


    //val tensorRDD = Tensor.readTensorBlock( tensorPath + "block/part-00000", tensorInfo.blockRank , rddPartitionSize)
    //val tensorRDD = Tensor.readTensorBlock( "data/donkey_spark_test/block/part-00000", tensorInfo.blockRank, rddPartitionSize)
    val tensorRDD = Tensor.readTensorBlock( "data/ConvertedData/block/part-00000", tensorInfo.blockRank, rddPartitionSize)

  tensorRDD.persist( MEMORY_AND_DISK )
    println("------------- OK 2 --------------------------------")

    println(" ======= BEGIN CLUSTERING ========")

    println(" ==== what are the IDS in ArraySeq[Int]")
    //tensorRDD.map(s => s._1).foreach(println)

    var tensorMat = tensorRDD.map(s => s._2).take(1)
    //tensorMat.foreach(println)

    //val fulltensor = tensor2Vects(matrixToRDD(tensorMat(0)), tensorInfo.tensorRank)
/*
  println("initial count " + tensorMat(0).numRows)
    var tens = localTensorUnfold( tensorMat(0), 0, tensorInfo.blockRank )
  println("final count  " + tens.size)
*/
  // get all 2D Matrices
  val RDDMatrix = blockTensorAsMatrices( tensorMat(0), tensorInfo.blockRank )

  println("OK")
  // Tranform matrices into RDD[Vector]
  val RDDVectors = blockTensorAsVectors(RDDMatrix)

  println("size array of rdd vectors = " + RDDVectors.count())

  println("------------------------------- CLUSTERING ------------------------ ")
  val test = new KMeansClustering(sc, 3, 5, tensorInfo.tensorDims)
  test.train(RDDVectors)

    // Create vectors in one block
    // we know that each block i of size 71*71*31*31
    //var newmatvect = tensorMat(0).toArray
    //matrixToRDD(tensorMat(0))
    //newmatvect.foreach(println)

    // a real 2D DenseMatrix
    //val newmat = new DMatrix(tensorInfo.blockRank(1), tensorInfo.blockRank(0), newmatvect.take(tensorInfo.blockRank(0) * tensorInfo.blockRank(1)))
    //println(newmat)

    // convert it to an RDD Vector
    //val finalvects = matrixToRDD(newmat)

    // convert linear to 4D coor
   //println(linear2Sub(180, tensorInfo.tensorRank))

    //val test = new KMeansClustering(sc, 3, 5, tensorInfo.tensorDims)
    //test.train(finalvects)

    /*finalvects.foreach(println)

    println(" ======= printing clusters centers ======= ")
    var clusters = KMeans.train(finalvects, 2, 10)
    clusters.clusterCenters.foreach(println)*/

    //var parsedData = tensorRDD.map(s => s._2) // get an RDD of DenseMatrices
    //parsedData.collect().foreach(println)


    // convert RDD to Dataframe
    //val mydmat = DMatrix(parsedData.collect())
    //val mydmat = parsedData.map(x => DMatrix(x))

  /*
    val mydmat: DMatrix = parsedData.first()
    //println(mydmat)
    var vectData = matrixToRDD(mydmat)
    //vectData.collect().foreach(println)
    println("printing data")
    //vectData.foreach(println)

    println(" ======= printing clusters centers ======= ")
    var clusters = KMeans.train(vectData, 5, 10)
    clusters.clusterCenters.foreach(println)

    println(" ======= END CLUSTERING ==========")
    */
}

    //println(tensorRDD)
    //tensorRDD.foreach(println)
    //println("number rows = " + tensorRDD.count())
    //return

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