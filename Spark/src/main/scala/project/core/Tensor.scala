package project.core

import java.nio.ByteBuffer

import org.apache.spark.{SparkConf, SparkContext}
import java.nio.file.{Files, Paths}
import breeze.linalg._
import org.apache.spark.mllib.linalg.{Vectors, DenseMatrix => DMatrix, Vector => MVector}

import scala.collection.{mutable => CM}
import org.apache.hadoop.io.{BytesWritable, LongWritable}
import org.apache.hadoop.mapred._
import Tensor.{convertBytes2Tuple, sc}
import org.apache.spark.rdd._
import org.apache.spark.broadcast.Broadcast
import javacode._
import breeze.numerics._



object Tensor {


  val conf = new SparkConf()
    .setAppName("CTA algorithm")
    .setMaster("local[*]")

  val sc = SparkContext.getOrCreate(conf)

  val sqlContext = new org.apache.spark.sql.SQLContext(sc)


  /*def setSparkContext( inSC: SparkContext  ): Unit =
  {
    sc = inSC
  }*/

  //-----------------------------------------------------------------------------------------------------------------
  // Class to store tensor information
  //-----------------------------------------------------------------------------------------------------------------
  class TensorInfo( inTensorDims: Int, inTensorRank: Array[Int], inBlockRank: Array[Int], inBlockNum: Array[Int],
                    inBlockFinal: Array[Int] ) extends Serializable
  {
    val tensorDims = inTensorDims
    val tensorRank = inTensorRank
    val blockRank = inBlockRank
    val blockNum = inBlockNum
    val blockFinal = inBlockFinal
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Initialize an empty tensor
  //-----------------------------------------------------------------------------------------------------------------
  def createNullTensorInfo(): TensorInfo =
  {
    new TensorInfo( 0, Array(0), Array(0), Array(0), Array(0) )
  }


  //-----------------------------------------------------------------------------------------------------------------
  // Convert bytes array to tuple2( blockSubIndex, DenseVector )
  //-----------------------------------------------------------------------------------------------------------------
  private def convertBytes2Tuple( bytesArray: Array[Byte], tensorDims: Int )
  : ( CM.ArraySeq[Int], DenseMatrix[Double] ) =
  {
    val byteBuffer = ByteBuffer.wrap( bytesArray )

    // Get block subindex
    //*val blockSubIndex = new Array[Int]( tensorDims )
    val blockSubIndex = new CM.ArraySeq[Int]( tensorDims )
    for( i <- 0 until tensorDims )
      blockSubIndex(i) = byteBuffer.getInt()

    // Get tensor block content
    val doubleBuffer = byteBuffer.asDoubleBuffer()
    val vectorSize = doubleBuffer.remaining()
    val doubleArray = new Array[Double]( vectorSize )
    for( i <- 0 until vectorSize  )
      doubleArray(i) = doubleBuffer.get()

    // Create a dense matrix and size is vectorSize*1
    //*val outVector = new DenseVector( doubleArray )
    val outMatrix = new DenseMatrix( vectorSize, 1, doubleArray )

    //*( blockSubIndex.toList, outMatrix )
    ( blockSubIndex, outMatrix )
  }


  //-----------------------------------------------------------------------------------------------------------------
  // Read tensor header
  // ------------------
  //
  //-----------------------------------------------------------------------------------------------------------------
  def readTensorHeader ( inPath: String ): TensorInfo =
  {
    println("File path exists : " + Files.exists(Paths.get(inPath)))

    val argArray = sc.textFile(inPath).collect()

    var tensorDims: Int = 0

    var line = argArray(0).split(" ")
    if ( line(0) == "TensorDims" )
    {
      tensorDims = line(1).toInt
    }
    else
    {
      println( "Tensor Header error!!" )
    }

    var tensorRank = new Array[Int]( tensorDims )
    var blockRank = new Array[Int]( tensorDims )
    var blockNum = new Array[Int]( tensorDims )

    for( i <- 1 to 3 )
    {
      line = argArray(i).split(" ")
      line(0) match
      {
        case "TensorRank" => tensorRank = line(1).split(",").map(_.toInt)
        case "BlockRank" => blockRank = line(1).split(",").map(_.toInt)
        case "BlockNumber" => blockNum = line(1).split(",").map(_.toInt)
      }
    }

    val blockFinal = new Array[Int]( tensorDims )
    for( i <- 0 until tensorDims )
      blockFinal(i) = ( (tensorRank(i)-1) % blockRank(i) ) + 1

    new TensorInfo( tensorDims, tensorRank, blockRank, blockNum, blockFinal )

  }





  //-----------------------------------------------------------------------------------------------------------------
  // Read tensor block
  //-----------------------------------------------------------------------------------------------------------------
  def readTensorBlock ( inPath: String, blockRanks: Array[Int] , rddPartitionSize: String): RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ] =
  {
    val tensorDims = blockRanks.length

    // Set Hadoop configuration
    val hadoopJobConf = new JobConf( sc.hadoopConfiguration )

    hadoopJobConf.set("mapred.min.split.size", rddPartitionSize)
    //val blockPath = inPath + "block/"
    val blockPath = inPath
    FileInputFormat.setInputPaths( hadoopJobConf, blockPath )
    val recordSize = ( blockRanks.product * 8 ) + ( tensorDims * 4 )
    MyFixedLengthInputFormat.setRecordLength( hadoopJobConf, recordSize )

    // Read tensor block data from HDFS and convert to tuple2( blockSubIndex, DenseVector ) format
    val bytesRdd = sc.hadoopRDD( hadoopJobConf, classOf[MyFixedLengthInputFormat], classOf[LongWritable],
      classOf[BytesWritable] )
    //*val blockRDD = bytesRdd.map( pair => pair._2.getBytes )
    //*                       .map{ case( bArray ) => convertBytes2Tuple( bArray, tensorDims ) }
    val blockRDD = bytesRdd.map{ case( _, bytes ) => convertBytes2Tuple( bytes.getBytes, tensorDims ) }

    blockRDD

  }


  //-----------------------------------------------------------------------------------------------------------------
  // Convert RDD[DenseMatrix] into RDD[Vector]
  //-----------------------------------------------------------------------------------------------------------------
 /* def matrixRDDToVectorRDD(m: RDD[DMatrix]) : RDD[MVector] = {
    var vect = null
    var nbMatrices = m.count()
    for(i <- 0 until nbMatrices){
      vect =
    }
  }*/

  def matrixToRDD(m: DMatrix): RDD[MVector] = {
    val columns = m.toArray.grouped(m.numRows)
    val rows = columns.toSeq.transpose
    val vectors = rows.map(row => Vectors.dense(row.toArray))
    sc.parallelize(vectors)
  }

}
