package project.core.v2

/**
  * Created by root on 2015/12/21.
  */
import java.nio._
import java.io.File
import java.io.PrintWriter

import scala.io.Source
import breeze.linalg._
import breeze.numerics._
import org.apache.hadoop.io.{BytesWritable, IntWritable, LongWritable}
import org.apache.hadoop.mapred._
import org.apache.spark.{Partitioner, SparkContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec
import scala.collection.{mutable => CM}
import project.core.javacode._
import org.apache.spark.mllib.linalg.{Vectors, DenseMatrix => DMatrix, Vector => MVector}
import breeze.linalg.{DenseMatrix => BMatrix}
import breeze.numerics._
import project.core.original.MySpark
import project.core.original.Tensor.{defineSubIndex, localTensorUnfoldBlock, permute, roundUp}

import scala.collection.immutable.{Vector => IVector}
import scala.{Vector => Vect}
object Tensor
{
  var sc: SparkContext = null

  def setSparkContext(inSC: SparkContext): Unit = {
    sc = inSC
    //    Tensor.setSparkContext( sc )
  }
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
  // Class to store basis matrices information
  //-----------------------------------------------------------------------------------------------------------------
  class BasisInfo( inNbRows: Array[Int], inNbCols: Array[Int] ) extends Serializable
  {
    val nbRows = inNbRows
    val nbCols = inNbCols
  }



  //-----------------------------------------------------------------------------------------------------------------
  // My partitioner to make one partition of RDD only have one tensor block
  //-----------------------------------------------------------------------------------------------------------------
  class MyPartitioner( blockNum: Array[Int] ) extends Partitioner
  {
    val partitions = blockNum.product
    require( partitions >= 0, s"Number of partitions ($partitions) cannot be negative." )

    def numPartitions: Int = partitions

    def getPartition( key: Any ): Int =
    {
      //*val keyList = key.asInstanceOf[ List[Int] ]
      val keySeq = key.asInstanceOf[ CM.ArraySeq[Int] ]
      val tensorDims = blockNum.length

      val cumArray = new Array[Int]( tensorDims )
      cumArray(0) = 1
      var result = 0
      result = result + cumArray(0) * keySeq(0)
      for( i <- 1 until tensorDims )
      {
        cumArray(i) = cumArray(i-1) * blockNum(i-1)
        result = result + cumArray(i) * keySeq(i)
      }

      result
    }
  }

  def createNullTensorInfo(): TensorInfo =
  {
    new TensorInfo( 0, Array(0), Array(0), Array(0), Array(0) )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Read tensor header
  //-----------------------------------------------------------------------------------------------------------------
  def readTensorHeader ( inPath: String ): TensorInfo =
  {
    val headerPath = inPath + "header/"
    val argArray = MySpark.sc.textFile( headerPath ).collect()
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
  // Read basis matrices header
  //-----------------------------------------------------------------------------------------------------------------
  def readBasisHeader ( inPath: String, tensorDims: Int): BasisInfo =
  {
    val headerPath = inPath + "header/"
    val argArray = MySpark.sc.textFile( headerPath ).collect()
    var line = argArray(0).split(" ")

    var nbRows = new Array[Int]( tensorDims )
    var nbCols = new Array[Int]( tensorDims )

    for( i <- 1 to 3 )
    {
      line(0) match
      {
        case "NbRows" => nbRows = line(1).split(",").map(_.toInt)
        case "NbCols" => nbCols = line(1).split(",").map(_.toInt)
      }
      line = argArray(i).split(" ")
    }
    new BasisInfo( nbRows, nbCols )

  }

  //-----------------------------------------------------------------------------------------------------------------
  // Read basis matrices
  //-----------------------------------------------------------------------------------------------------------------
  def readBasisMatrices ( inPath: String, blockRanks: Array[Int]): Array[BMatrix[Double]] =
  {
    val tensorDims = blockRanks.length

    // Set Hadoop configuration
    val hadoopJobConf = new JobConf( MySpark.sc.hadoopConfiguration )

    val blockRanks2 = Array(71,71, 31,31)
    val recordSize = ( blockRanks2.product * 8 ) + ( tensorDims * 4 )

    hadoopJobConf.set("mapred.min.split.size", recordSize.toString)

    val blockPath = inPath
    FileInputFormat.setInputPaths( hadoopJobConf, blockPath)
    MyFixedLengthInputFormat.setRecordLength( hadoopJobConf, recordSize )

    // Read tensor block data from HDFS and convert to tuple2( blockSubIndex, DenseVector ) format
    val bytesRdd = MySpark.sc.hadoopRDD( hadoopJobConf, classOf[MyFixedLengthInputFormat], classOf[LongWritable],
      classOf[BytesWritable] )

    val blockRDD = bytesRdd.map{ case( _, bytes ) => convertBytes2Matrix( bytes.getBytes, tensorDims ) }

    blockRDD.map{ x => x._2.reshape(x._1, x._2.rows / x._1)}.collect().reverse
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Read tensor block
  //-----------------------------------------------------------------------------------------------------------------
  def readTensorBlock ( inPath: String, blockRanks: Array[Int]): RDD[ (CM.ArraySeq[Int], DMatrix) ] =
  {
    val tensorDims = blockRanks.length

    // Set Hadoop configuration
    val hadoopJobConf = new JobConf( MySpark.sc.hadoopConfiguration )

    val recordSize = ( blockRanks.product * 8 ) + ( tensorDims * 4 )

    hadoopJobConf.set("mapred.min.split.size", recordSize.toString)

    val blockPath = inPath + "block/"
    FileInputFormat.setInputPaths( hadoopJobConf, blockPath)
    MyFixedLengthInputFormat.setRecordLength( hadoopJobConf, recordSize )

    // Read tensor block data from HDFS and convert to tuple2( blockSubIndex, DenseVector ) format
    val bytesRdd = MySpark.sc.hadoopRDD( hadoopJobConf, classOf[MyFixedLengthInputFormat], classOf[LongWritable],
      classOf[BytesWritable] )

    val blockRDD = bytesRdd.map{ case( _, bytes ) => convertBytes2Tuple( bytes.getBytes, tensorDims ) }

    blockRDD

  }


  //-----------------------------------------------------------------------------------------------------------------
  // Read tensor block
  //-----------------------------------------------------------------------------------------------------------------
  def readClusterBlock ( inPath: String, blockRanks: Array[Int]): RDD[ Array[(CM.ArraySeq[Int], DMatrix)] ] =
  {
    val tensorDims = blockRanks.length

    // Set Hadoop configuration
    val hadoopJobConf = new JobConf( MySpark.sc.hadoopConfiguration )

    val recordSize = ( blockRanks.product * 8 ) + ( tensorDims * 4 )

    hadoopJobConf.set("mapred.min.split.size", recordSize.toString)

    val blockPath = inPath + "block/"
    FileInputFormat.setInputPaths( hadoopJobConf, blockPath)
    MyFixedLengthInputFormat.setRecordLength( hadoopJobConf, recordSize )

    // Read tensor block data from HDFS and convert to tuple2( blockSubIndex, DenseVector ) format
    val bytesRdd = MySpark.sc.hadoopRDD( hadoopJobConf, classOf[MyFixedLengthInputFormat], classOf[LongWritable],
      classOf[BytesWritable] )

    val blockRDD = bytesRdd.map{ case( _, bytes ) => convertBytes2Tuple( bytes.getBytes, tensorDims ) }

    MySpark.sc.parallelize(Seq(blockRDD.collect()))

  }

  //-----------------------------------------------------------------------------------------------------------------
  // Save basis matrices header
  //-----------------------------------------------------------------------------------------------------------------
  def saveBasisHeader ( inPath: String, basisInfo: BasisInfo ) =
  {
    // Set save path
    val savePath = inPath + "header/"

    //
    val infoArray = new Array[ String ](4)
    infoArray(0) = "NbRows" + " " + basisInfo.nbRows.toString
    infoArray(1) = "NbCols" + " " + basisInfo.nbCols.mkString(",")
    MySpark.sc.parallelize( infoArray, 1 ).saveAsTextFile( savePath )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Save tensor header
  //-----------------------------------------------------------------------------------------------------------------
  def saveTensorHeader ( inPath: String, tensorInfo: TensorInfo ) =
  {
    // Set save path
    //val filePart = inPath.split( "/" )
    val savePath = inPath + "header/"

    //
    val infoArray = new Array[ String ](4)
    infoArray(0) = "TensorDims" + " " + tensorInfo.tensorDims.toString
    infoArray(1) = "TensorRank" + " " + tensorInfo.tensorRank.mkString(",")
    infoArray(2) = "BlockRank" + " " + tensorInfo.blockRank.mkString(",")
    infoArray(3) = "BlockNumber" + " " + tensorInfo.blockNum.mkString(",")
    MySpark.sc.parallelize( infoArray, 1 ).saveAsTextFile( savePath )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Save tensor block
  //-----------------------------------------------------------------------------------------------------------------
  def saveTensorBlock ( tensorPath: String, tensorRDD: RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ] ) =
  {
    // Set Hadoop configuration
    val hadoopJobConf = new JobConf( MySpark.sc.hadoopConfiguration )
    val savePath = tensorPath + "block/"

    val saveRDD = tensorRDD.map{ case( blockSubIndex, tensorVector ) =>
      convertTuple2Bytes( blockSubIndex, tensorVector ) }

    saveRDD.saveAsHadoopFile( savePath, classOf[BytesWritable], classOf[BytesWritable], classOf[BinaryOutputFormat],
      hadoopJobConf, None )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Reshape a block with cols and rows
  //-----------------------------------------------------------------------------------------------------------------
  def reshapeBlock(index: CM.ArraySeq[Int], dim: Int, info: TensorInfo, block: BMatrix[Double]): BMatrix[Double] ={
    val size = index.zipWithIndex.map{ x => if(x._1 + 1 == info.blockNum(x._2)) info.blockFinal(x._2) else info.blockRank(x._2)}
    val odims = size.zipWithIndex.filter{ _._2 != dim }.map{ _._1 }
    block.reshape(size(dim), odims.product)
  }


  //-----------------------------------------------------------------------------------------------------------------
  // Save tensor block
  //-----------------------------------------------------------------------------------------------------------------
  def saveTmpBlock ( ) =
  {
    val myArray = Array[Double](1.0, 2.0, 4.0, 5.0, 1.0, 1.0, 3.0, 4.0)
    var mat = new DenseMatrix[Double](4, 2, myArray)
    val tensorRDD = MySpark.sc.parallelize(Seq(mat))
    // Set Hadoop configuration
    val hadoopJobConf = new JobConf( MySpark.sc.hadoopConfiguration )
    val savePath = "data/block/"

    val saveRDD = tensorRDD.map{ case( tensorVector ) =>
      convertTuple2Bytes( tensorVector ) }

    saveRDD.saveAsTextFile(savePath)
    //saveRDD.saveAsHadoopFile( savePath, classOf[BytesWritable], classOf[BytesWritable], classOf[BinaryOutputFormat],
    // hadoopJobConf, None )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Save Basis matrices
  //-----------------------------------------------------------------------------------------------------------------
  def saveBasisMatrices ( basisPath: String, bcBasisMatrixArray: Array[ Broadcast[DenseMatrix[Double]] ],
                          dimSeq: Array[Int] ) =
  {
    // Set Hadoop configuration
    val hadoopJobConf = new JobConf( MySpark.sc.hadoopConfiguration )
    val savePath = basisPath

    // Get basis matrices from broadcast variable and convert it into bytes array
    val matrixNum = bcBasisMatrixArray.length
    //val bytesArray = new Array[ ( BytesWritable, BytesWritable ) ]( matrixSize )
    val matrixArray = new Array[ ( Int, DenseMatrix[Double] ) ]( matrixNum )

    for( i <- 0 until matrixNum )
    {
      if( dimSeq.contains( i ) )
        matrixArray( i ) = ( i, bcBasisMatrixArray(i).value )
      else
        matrixArray( i ) = ( 1, DenseMatrix.fill(1,1){0.0} )
    }

    // Parallelize bytes array and write to HDFS
    //*val saveRDD = MySpark.sc.parallelize( bytesArray )
    val saveRDD = MySpark.sc.parallelize( matrixArray, matrixNum )
      .map{ case( dim, matrix ) => convertMatrix2Bytes( matrix.rows, matrix ) }
    //saveRDD.saveAsTextFile(savePath)
    saveRDD.saveAsHadoopFile( savePath, classOf[BytesWritable], classOf[BytesWritable], classOf[BinaryOutputFormat],
      hadoopJobConf, None )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Perform mode-N products
  //-----------------------------------------------------------------------------------------------------------------
  def modeNProducts( tensorRDD: RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ], tensorInfo: TensorInfo,
                     prodsSeq: Array[Int], bcMatrixArray: Array[Broadcast[DenseMatrix[Double]]] )
  : ( RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ], TensorInfo )  =
  {
    // Check how many dimension to do mode-n products
    val prodsDims = prodsSeq.length
    //    val test = tensorRDD.join(tensorRDD).collect()

    // Tail recursive control flow to implement mode-n products
    @tailrec
    def recursiveCall( tensorRDD: RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ], tensorInfo: TensorInfo, iter: Int )
    : ( RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ], TensorInfo )  =
    {
      if( iter == prodsDims )
      {
        //*( tensorRDD.reduceByKey( new MyPartitioner( tensorInfo.blockNum ), ( a, b ) => a + b ), tensorInfo )
        ( tensorRDD, tensorInfo )
      }
      else
      {
        val ( prodRDD, prodTensorInfo ) =
          modeNProduct( tensorRDD, tensorInfo, prodsSeq(iter), bcMatrixArray( prodsSeq(iter) ) )
        recursiveCall( prodRDD, prodTensorInfo, iter + 1 )
      }
    }

    // Return mode-N products tensorRDD
    recursiveCall( tensorRDD, tensorInfo, 0 )

  }

  //-----------------------------------------------------------------------------------------------------------------
  // Perform mode-N products
  //-----------------------------------------------------------------------------------------------------------------
  def modeNProducts2( tensorRDD: RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ], tensorInfo: TensorInfo,
                      prodsSeq: Array[Int], bcMatrixArray: Array[Broadcast[DenseMatrix[Double]]] )
  : ( RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ], TensorInfo )  =
  {
    // Set prods tensor information
    val prodsTensorRank = tensorInfo.tensorRank.clone()
    val prodsBlockNum = tensorInfo.blockNum.clone()
    val prodsBlockFinal = tensorInfo.blockFinal.clone()
    for( prodDim <- prodsSeq )
    {
      val prodDimRank = bcMatrixArray( prodDim ).value.cols
      val prodDimBlockRank = tensorInfo.blockRank( prodDim )
      prodsTensorRank( prodDim ) = prodDimRank
      prodsBlockNum( prodDim ) = ceil( prodDimRank.toFloat / prodDimBlockRank.toFloat ).toInt
      prodsBlockFinal( prodDim ) = ( ( prodDimRank - 1 ) % prodDimBlockRank ) + 1
    }
    val prodsInfo = new TensorInfo( tensorInfo.tensorDims, prodsTensorRank, tensorInfo.blockRank, prodsBlockNum,
      prodsBlockFinal )

    val prodsRDD = tensorRDD.flatMap{ case( blockSubIndex, denseMatrix ) =>
      localMultiDimMMMany( blockSubIndex, denseMatrix, prodsSeq, bcMatrixArray,
        tensorInfo.blockRank, tensorInfo.blockNum,
        tensorInfo.blockFinal, prodsInfo.tensorRank,
        prodsInfo.blockNum ) }
      .reduceByKey( new MyPartitioner( prodsInfo.blockNum ), ( a, b ) => a + b )

    ( prodsRDD, prodsInfo )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Perform mode-N product
  //-----------------------------------------------------------------------------------------------------------------
  def modeNProduct( tensorRDD: RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ], tensorInfo: TensorInfo, prodDim: Int,
                    bcMatrix: Broadcast[DenseMatrix[Double]] )
  :( RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ], TensorInfo )  =
  {
    // Reset tensor information
    val prodDimRank = bcMatrix.value.cols
    val prodTensorRank = tensorInfo.tensorRank.clone()
    prodTensorRank( prodDim ) = prodDimRank

    val prodDimBlockRank = tensorInfo.blockRank( prodDim )
    val prodBlockNum = tensorInfo.blockNum.clone
    prodBlockNum( prodDim ) = ceil( prodDimRank.toFloat / prodDimBlockRank.toFloat ).toInt

    val prodBlockFinal = tensorInfo.blockFinal.clone()
    prodBlockFinal( prodDim ) = ( ( prodDimRank - 1 ) % prodDimBlockRank ) + 1

   val outTensorInfo = new TensorInfo( tensorInfo.tensorDims, prodTensorRank, tensorInfo.blockRank, prodBlockNum,
      prodBlockFinal )

    // Call map function-localMatrixMultiplyMany and reduceByKey to finish mode-N product
    //*val prodRDD = tensorRDD.flatMap{ case( blockSubIndex, denseVector ) =>
    //*  localMatrixMultiplyMany( blockSubIndex, denseVector, prodDim, bcMatrix,
    //*    tensorInfo.blockRank, tensorInfo.blockNum, tensorInfo.blockFinal ) }
    //*  .reduceByKey( new MyPartitioner( outTensorInfo.blockNum ), ( a, b ) => a + b )

    val prodRDD = tensorRDD.flatMap{ case( blockSubIndex, denseMatrix ) =>
      localMMMany( blockSubIndex, denseMatrix, prodDim, bcMatrix, tensorInfo.blockRank,
        tensorInfo.blockNum, tensorInfo.blockFinal ) }
      .reduceByKey( new MyPartitioner( outTensorInfo.blockNum ), ( a, b ) => a + b )

    ( prodRDD , outTensorInfo )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Perform extract basis
  //-----------------------------------------------------------------------------------------------------------------
  def extractBasis( tensorRDD: RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ], tensorInfo: TensorInfo, extDim: Int,
                    extRank: Int )
  : DenseMatrix[Double] =
  {
    val covMatrix = getCovMatrix( tensorRDD, tensorInfo, extDim )

    val eigPair = eigSym( covMatrix )
    var basisMatrix = DenseMatrix.zeros[Double]( covMatrix.rows, extRank)
    var eigenvalues = DenseVector.zeros[Double](extRank)
    if(covMatrix.rows < extRank){
      for(i <- 0 until covMatrix.rows){
        basisMatrix( ::, i) := eigPair.eigenvectors( ::, i )
        eigenvalues(i) = eigPair.eigenvalues(i)
      }
      //add missing columns
      //basisMatrix = DenseMatrix.horzcat(basisMatrix, DenseMatrix.zeros[Double](covMatrix.rows, extRank - covMatrix.cols))

    } else {
      for (i <- 0 until extRank) {
        basisMatrix( ::, i ) := eigPair.eigenvectors( ::, i )
        eigenvalues(i) = eigPair.eigenvalues(i)
      }
    }

    var eigenVal = sqrt(DenseVector((eigenvalues).toArray.sortWith( _ > _)))


    basisMatrix
  }


  //-----------------------------------------------------------------------------------------------------------------
  // Compute covariance matrix for debug
  //-----------------------------------------------------------------------------------------------------------------
  def getCovMatrix( tensorRDD: RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ], tensorInfo: TensorInfo, extDim: Int )
  : DenseMatrix[Double] =
  {
    // Reset key and group
    val tempBlockNum = tensorInfo.blockNum.clone()
    tempBlockNum( extDim ) = 1
    var all = tensorRDD.map{ case(blockSubIndex, tensorVector) => tensorVector}.collect()
    val groupRDD = tensorRDD.map{ case( blockSubIndex, tensorVector ) =>
      extractBasisResetKey( blockSubIndex, tensorVector, extDim ) }
      .groupByKey( new MyPartitioner( tempBlockNum )  )

    // Compute partial covariance matrix each group and than summation
    //*val covRDD = groupRDD.map{ case( a, b ) => computePartialCovMatrix( a, b, tensorInfo, extDim ) }
    //*val vectorRDD = covRDD.flatMap{ case( a ) => tearCovMatrixIntoVector( a ) }
    //*                      .reduceByKey( ( a, b ) => a + b )
    //*                      .map{ case( a, b ) => Vectors.dense( b.toArray ) }
    val covMatrix = groupRDD.map{ case( a, b ) => computePartialCovMatrix( a, b, tensorInfo, extDim ) }
      .reduce( ( a, b ) => a + b )

    covMatrix
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Project onto matrix
  //-----------------------------------------------------------------------------------------------------------------
  def projectOntoMatrix(matrixA: DenseMatrix[Double], matrixB: DenseMatrix[Double], TransFlagA: Boolean, TransFlagB: Boolean): DenseMatrix[Double] ={

    var transMatA = matrixA
    var transMatB = matrixB
    var transSize = matrixB.rows
    var projectionMat: DenseMatrix[Double] = DenseMatrix.zeros[Double](transSize, transSize)

    if(TransFlagA && matrixB.cols != matrixA.rows) {
      transMatA = matrixA.t
    }

    projectionMat = transMatB * transMatA

    projectionMat
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Project onto matrix
  //-----------------------------------------------------------------------------------------------------------------
  def projectOntoMatrix(matrixA: Array[DenseMatrix[Double]], matAInfo: Tensor.TensorInfo, matrixB: DenseMatrix[Double], TransFlagA: Boolean, TransFlagB: Boolean): DenseMatrix[Double] ={

    var transMatA = matrixA
    var transMatB = matrixB
    var transSize = matrixB.rows
    var projectionMat: DenseMatrix[Double] = DenseMatrix.zeros[Double](transSize, transSize)

    if(TransFlagA) {
      transMatA = matrixA.map { x => x.t }
    }

    for(i <- 0 to matAInfo.blockNum.product){
      if(i == 0){
        projectionMat = transMatB * transMatA.apply(0)
      } else{
        projectionMat = projectionMat + transMatB * transMatA.apply(i)
      }
    }

    projectionMat
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Convert matrix into binary format
  //-----------------------------------------------------------------------------------------------------------------
  private def convertMatrix2Bytes( dim: Int, matrix: DenseMatrix[Double] ): ( BytesWritable, BytesWritable ) =
  {
    // Set key bytebuffer to store dim, cols and rows of matrix
    val keyBuffer = ByteBuffer.allocate( 4 * 3 )
    keyBuffer.asIntBuffer().put( dim ).put( matrix.rows ).put( matrix.cols )

    // Set value bytebuffer and put matrix
    val valueBuffer = ByteBuffer.allocate( 8 * matrix.size )
    valueBuffer.asDoubleBuffer().put( matrix.data )

    ( new BytesWritable( keyBuffer.array() ), new BytesWritable( valueBuffer.array() ) )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Convert bytes array to basisMatrix
  //-----------------------------------------------------------------------------------------------------------------
  private def convertBytes2Matrix( bytesArray: Array[Byte], tensorDims: Int )
  : (Int, BMatrix[Double]) =
  {
    val byteBuffer = ByteBuffer.wrap( bytesArray )

    // Get block subindex
    val dim: Array[Int] = new Array[Int](1)
    dim(0) = byteBuffer.getInt()

    // Get tensor block content
    val doubleBuffer = byteBuffer.asDoubleBuffer()
    val vectorSize = doubleBuffer.remaining()
    val doubleArray = new Array[Double]( vectorSize )
    for( i <- 0 until vectorSize - 1)
      doubleArray(i) = doubleBuffer.get()

    // Create a dense matrix and size is vectorSize*1
    val outMatrix = new BMatrix( vectorSize - 1, 1, doubleArray )

    (dim(0), outMatrix)
  }


  //-----------------------------------------------------------------------------------------------------------------
  // Convert bytes array to tuple2( blockSubIndex, DenseVector )
  //-----------------------------------------------------------------------------------------------------------------
  private def convertBytes2Tuple( bytesArray: Array[Byte], tensorDims: Int )
  : ( CM.ArraySeq[Int], DMatrix ) =
  {
    val byteBuffer = ByteBuffer.wrap( bytesArray )

    // Get block subindex
    val blockSubIndex = new CM.ArraySeq[Int]( tensorDims )
    for( i <- 0 until tensorDims )
      blockSubIndex(i) = byteBuffer.getInt()

    // Get tensor block content
    val doubleBuffer = byteBuffer.asDoubleBuffer()
    val vectorSize = doubleBuffer.remaining()
    val doubleArray = new Array[Double]( vectorSize )
    for( i <- 0 until vectorSize)
      doubleArray(i) = doubleBuffer.get()

    // Create a dense matrix and size is vectorSize*1
    val outMatrix = new DMatrix( vectorSize, 1, doubleArray )

    ( blockSubIndex, outMatrix )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Convert tuple2( blockSubIndex, DenseVector ) tp bytes array
  //-----------------------------------------------------------------------------------------------------------------
  def convertTuple2Bytes( blockSubIndex: CM.ArraySeq[Int], tensorMatrix: DenseMatrix[Double] )
  : ( BytesWritable, BytesWritable ) =
  {
    // Set key bytebuffer and put block subindex
    val keyBuffer = ByteBuffer.allocate( 4 * blockSubIndex.length )
    keyBuffer.asIntBuffer().put( blockSubIndex.toArray )

    // Set value bytebuffer and put tensor vector
    //*val valueBuffer = ByteBuffer.allocate( 8 * tensorVector.length )
    //*valueBuffer.asDoubleBuffer().put( tensorVector.data )
    val valueBuffer = ByteBuffer.allocate( 8 * tensorMatrix.size )
    valueBuffer.asDoubleBuffer().put( tensorMatrix.data )

    ( new BytesWritable( keyBuffer.array() ), new BytesWritable( valueBuffer.array() ) )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Convert tuple2( blockSubIndex, DenseVector ) tp bytes array
  //-----------------------------------------------------------------------------------------------------------------
  def convertTuple2Bytes( tensorMatrix: DenseMatrix[Double] )
  : ( BytesWritable, BytesWritable ) =
  {
    var blockSubIndex = CM.ArraySeq(0,0)
    // Set key bytebuffer and put block subindex
    val keyBuffer = ByteBuffer.allocate( 4 * blockSubIndex.length )
    keyBuffer.asIntBuffer().put( blockSubIndex.toArray )

    // Set value bytebuffer and put tensor vector
    //*val valueBuffer = ByteBuffer.allocate( 8 * tensorVector.length )
    //*valueBuffer.asDoubleBuffer().put( tensorVector.data )
    val valueBuffer = ByteBuffer.allocate( 8 * tensorMatrix.size )
    valueBuffer.asDoubleBuffer().put( tensorMatrix.data )

    ( new BytesWritable( keyBuffer.array() ), new BytesWritable( valueBuffer.array() ) )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Perform local tensor unfold
  //-----------------------------------------------------------------------------------------------------------------
  def localTensorUnfoldBlock( tensorMatrixTmp: DMatrix, ids: CM.ArraySeq[Int], unfoldDim: Int, blockRank: Array[Int], blockNum: Array[Int], tensorRank: Array[Int] )
  : BMatrix[Double] =
  {
    /*for(l <- 0 until tensorMatrixTmp.numRows){
     if(tensorMatrixTmp.apply(l, 0) == 0.283203125){
       println(" l = " + l)
     }

    }*/
    var modifiedRank: Int = 0
    val tensorDims = blockRank.length
    var unfoldMatrix: BMatrix[Double] = null
    val unfoldRank = new Array[Int](2)
    val tensorMatrix = new BMatrix[Double](tensorMatrixTmp.numRows, tensorMatrixTmp.numCols, tensorMatrixTmp.values)
    if( unfoldDim == 0 )                   // For first dimension
    {
      unfoldRank(0) = blockRank(0)
      unfoldRank(1) = 1
      for( i <- 1 until tensorDims)
      {
        if(ids(i) == (blockNum(i) - 1) && blockRank(i) % tensorRank(i) != 0)
          modifiedRank = tensorRank(i) - (blockRank(i) * (blockNum(i) - 1))
        else
          modifiedRank = blockRank(i)

        unfoldRank(1) = unfoldRank(1) * modifiedRank
      }

      if(ids(0) == (blockNum(0) - 1) && blockRank(0) % tensorRank(0) != 0)
        unfoldRank(0) = tensorRank(0) - (blockRank(0) * (blockNum(0) - 1))

      unfoldMatrix = tensorMatrix.reshape( unfoldRank(0), unfoldRank(1) )
    }
    else if( unfoldDim == tensorDims - 1 ) // For last dimension
    {
      println("ids : " + ids.toList + " " + tensorMatrixTmp.numRows)
      unfoldRank(0) = 1
      unfoldRank(1) = blockRank(unfoldDim)
      for( i <- 0 to tensorDims - 2 )
      {
        if(ids(i) == (blockNum(i) - 1) && blockRank(i) % tensorRank(i) != 0)
          modifiedRank = tensorRank(i) - (blockRank(i) * (blockNum(i) - 1))
        else
          modifiedRank = blockRank(i)
        unfoldRank(0) = unfoldRank(0) * modifiedRank
      }
      if(ids(unfoldDim) == (blockNum(tensorDims - 1 ) - 1) && blockRank(tensorDims - 1 ) % tensorRank(tensorDims - 1 ) != 0)
        unfoldRank(1) = tensorRank(tensorDims - 1 ) - (blockRank(tensorDims - 1 ) * (blockNum(tensorDims - 1 ) - 1))


      unfoldMatrix = tensorMatrix.reshape( unfoldRank(0), unfoldRank(1), false ).t
      /*for(w <- 0 until unfoldMatrix.rows)
        //println("(0, 34999)" + unfoldMatrix.apply(w,34999))
     for(w <- 0 until unfoldMatrix.rows){
        for(v <- 0 until unfoldMatrix.cols){
          //0.042442321777344
          if(unfoldMatrix.apply(w, v) == 0.283203125){
            println(ids)
            println("(" + w + "," + v + ") = " + unfoldMatrix.apply(w, v))
          }
        }
      }

      println("test")*/
    }
    else                                   // For other dimensions
    {
      println("ids : " + ids.toList + " " + tensorMatrixTmp.numRows)
      unfoldRank(0) = 1
      unfoldRank(1) = 1
      for( i <- 0 until tensorDims )
      {
        if(ids(i) == (blockNum(i) - 1) && blockRank(i) % tensorRank(i) != 0)
          modifiedRank = tensorRank(i) - (blockRank(i) * (blockNum(i) - 1))
        else
          modifiedRank = blockRank(i)

        if(i < unfoldDim)
          unfoldRank(0) = unfoldRank(0) * modifiedRank
        else
          unfoldRank(1) = unfoldRank(1) * modifiedRank

      }
      val temp = blockRank.product / blockRank(unfoldDim)

      unfoldMatrix = tensorMatrix.reshape( unfoldRank(0), unfoldRank(1), false ).t
        .reshape(blockRank(unfoldDim), temp, false)
    }

    unfoldMatrix
  }




  //-----------------------------------------------------------------------------------------------------------------
  // Perform local tensor unfold
  //-----------------------------------------------------------------------------------------------------------------
  def localTensorUnfoldBlock( tensorMatrix: BMatrix[Double], ids: CM.ArraySeq[Int], unfoldDim: Int, blockRank: Array[Int], blockNum: Array[Int], tensorRank: Array[Int] )
  : BMatrix[Double] =
  {
    var modifiedRank: Int = 0
    val tensorDims = blockRank.length
    var unfoldMatrix: BMatrix[Double] = null
    val unfoldRank = new Array[Int](2)
    if( unfoldDim == 0 )                   // For first dimension
    {
      unfoldRank(0) = blockRank(0)
      unfoldRank(1) = 1
      for( i <- 1 until tensorDims)
      {
        if(ids(i) == (blockNum(i) - 1) && blockRank(i) % tensorRank(i) != 0)
          modifiedRank = tensorRank(i) - (blockRank(i) * (blockNum(i) - 1))
        else
          modifiedRank = blockRank(i)

        unfoldRank(1) = unfoldRank(1) * modifiedRank
      }

      if(ids(0) == (blockNum(0) - 1) && blockRank(0) % tensorRank(0) != 0)
        unfoldRank(0) = tensorRank(0) - (blockRank(0) * (blockNum(0) - 1))

      unfoldMatrix = tensorMatrix.reshape( unfoldRank(0), unfoldRank(1) )
    }
    else if( unfoldDim == tensorDims - 1 ) // For last dimension
    {
      println("ids : " + ids.toList + " " + tensorMatrix.rows)
      unfoldRank(0) = 1
      unfoldRank(1) = blockRank(unfoldDim)
      for( i <- 0 to tensorDims - 2 )
      {
        if(ids(i) == (blockNum(i) - 1) && blockRank(i) % tensorRank(i) != 0)
          modifiedRank = tensorRank(i) - (blockRank(i) * (blockNum(i) - 1))
        else
          modifiedRank = blockRank(i)
        unfoldRank(0) = unfoldRank(0) * modifiedRank
      }
      if(ids(unfoldDim) == (blockNum(tensorDims - 1 ) - 1) && blockRank(tensorDims - 1 ) % tensorRank(tensorDims - 1 ) != 0)
        unfoldRank(1) = tensorRank(tensorDims - 1 ) - (blockRank(tensorDims - 1 ) * (blockNum(tensorDims - 1 ) - 1))


      unfoldMatrix = tensorMatrix.reshape( unfoldRank(0), unfoldRank(1), false ).t
    }
    else                                   // For other dimensions
    {
      unfoldRank(0) = 1
      unfoldRank(1) = 1
      for( i <- 0 until tensorDims )
      {
        if(ids(i) == (blockNum(i) - 1) && blockRank(i) % tensorRank(i) != 0)
          modifiedRank = tensorRank(i) - (blockRank(i) * (blockNum(i) - 1))
        else
          modifiedRank = blockRank(i)

        if(i < unfoldDim)
          unfoldRank(0) = unfoldRank(0) * modifiedRank
        else
          unfoldRank(1) = unfoldRank(1) * modifiedRank

      }
      val temp = blockRank.product / blockRank(unfoldDim)

      unfoldMatrix = tensorMatrix.reshape( unfoldRank(0), unfoldRank(1), false ).t
        .reshape(blockRank(unfoldDim), temp, false)
    }

    unfoldMatrix
  }

  def permute[T](i: Int, list: List[T]): List[T] = i match {
    case i if i > 0 => list match {
      case head :: xs => permute(i-1, xs ::: List(head))
      case Nil => permute(i-1, Nil)
    }
    case i if i < 0 => list.reverse match {
      case head :: xs => permute(i + 1, head :: xs.reverse)
      case Nil => permute(i+1, Nil)
    }
    case _ => list
  }

  def roundUp(d: Double) = math.ceil(d).toInt

  def defineSubIndex(IndexArray: Array[CM.ArraySeq[Int]], blockNum: Array[Int], unfoldDim: Int): Array[CM.ArraySeq[Int]] ={
    val nbDims = blockNum.length
    blockNum(unfoldDim) = 1
    val nbIds = blockNum.product

    var sortCol = unfoldDim
    var arr = IndexArray
    for(i <- 0 until nbDims){
      sortCol = (sortCol + 1) % 4
      /* if(sortCol < 0)
         sortCol = nbDims + sortCol*/
      arr = arr.map{ case(x) => x}.sortBy(x => (x(sortCol)))
    }

    arr
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Perform local tensor unfold
  //-----------------------------------------------------------------------------------------------------------------
  def localTensorUnfoldOriginal( tensorRDD: RDD[(CM.ArraySeq[Int], DMatrix)], unfoldDim: Int, tensorRank: Array[Int], blockRank: Array[Int], blockNumTmp: Array[Int])
  //: DenseMatrix[Double] =
  =
  {
    val tensorDims = tensorRank.length
    val unfoldRank: Array[Int] = Array(tensorRank(unfoldDim), tensorRank.product / tensorRank(unfoldDim))
    val unfoldBlockRank: Array[Int] = Array(blockRank(unfoldDim), blockRank.product / blockRank(unfoldDim))

    val unfoldSeq: Array[Int] = unfoldRank.zip(unfoldBlockRank).map{ case(x, y) => roundUp(x.toDouble / y.toDouble)}

    val totalNumBlocks = blockNumTmp.product / blockNumTmp(unfoldDim)

    //tensorRDD.map{ case(x, y) => (x(0), (x(1), x(2), x(3)))}.sortBy(_._1, true).map{ case(b, c) => (b, c._1, c._2, c._3)}.foreach(println)
    //tensorRDD.sortBy{ case(x) => x._1(unfoldDim)}.foreach(println)
    //tensorRDD.map{ case(x, y) => (x(0), (x(1), x(2), x(3)))}.sortBy(_._1, true).foreach(println)
    //tensorRDD.map{ case(x, y) => (x(0), (x(1), x(2), x(3)))}.groupBy(_._1).mapValues{_.map{ case(_, c) => c}}.map{ case(x, c) => c.map{z => (x, z._1, z._2, z._3)}}.flatMap(x => x).foreach(println)
    //map{ case(x, y) => x -> y.map{ case(_, (c)) => c}}.foreach(println)

    var subIndex = defineSubIndex(tensorRDD.map{ case(x, y) => x}.collect(), blockNumTmp.clone(), unfoldDim)
    // permuting block numbers depending on unfolding dimension
    val blockNum = permute(unfoldDim, blockNumTmp.toList).toArray

    var newBlocks: Array[(CM.ArraySeq[Int], BMatrix[Double])] = new Array[(CM.ArraySeq[Int], BMatrix[Double])](unfoldSeq.product + unfoldSeq(0))

    for( i <- 0 until unfoldSeq(0)){
      // Valid size of the block
      var validSize = min(unfoldRank(1), unfoldBlockRank(1))
      var unfoldSize = 0
      var blockIndexCol = 1
      //println("valid size = " + validSize)
      // initialize tensor
      var unfoldBlock: BMatrix[Double] = null

      for( j <- 0 until totalNumBlocks){
        var block = tensorRDD.filter{ case (x, y) => x == subIndex(j + (i * totalNumBlocks))}.map{ case(x, y) => localTensorUnfoldBlock(y, x, unfoldDim, blockRank, blockNumTmp, tensorRank)}.take(1)
        // get the size of the block
        var blockSize = block(0).cols

        unfoldSize += blockSize


        //println(subIndex(j + (i * totalNumBlocks)) + " => block size = " + block(0).rows + " * " + blockSize)

        if(unfoldSize >= validSize){
          unfoldSize -= validSize
          var catSize = blockSize - unfoldSize
          // reshape matrix into cat size -- concatenate with previous data
          if(unfoldBlock == null)
            unfoldBlock = block(0)(::, 0 to (catSize - 1))
          else
            unfoldBlock = BMatrix.horzcat(unfoldBlock, block(0)(::, 0 to (catSize - 1)))

          // Generation of new block is complete
          // Save the new block
          // HDFS
          newBlocks(i * unfoldSeq(1) + blockIndexCol) = (CM.ArraySeq[Int](i, blockIndexCol), unfoldBlock)

          // Update block index and size
          validSize = min(unfoldRank(1) - blockIndexCol * unfoldBlockRank(1), unfoldBlockRank(1))
          blockIndexCol += 1

          // Update the unfolded tensor (data left)
          unfoldBlock = block(0)(::, catSize to -1)

        } else {
          if(unfoldBlock == null)
            unfoldBlock = block(0)
          else {

            unfoldBlock = BMatrix.horzcat(unfoldBlock, block(0))
          }
        }

      }
    }

    println("unfoldRank")
    println(unfoldRank.toList)

    println("unfoldBlockRank")
    println(unfoldBlockRank.toList)

    println("unfoldSeq")
    println(unfoldSeq.toList)

    println("total of output blocks : " + totalNumBlocks)

    println("permuted block num")
    println(blockNum.toList)
  }
  /*
    def blockTensorAsMatrices(linearTensor: DMatrix, tensorRank: Array[Int]): RDD[DenseMatrix[Double]] ={
      val tensorDims = tensorRank.length
      var unfoldTensor: RDD[Vector[Double]] = null
      val unfoldRank = new Array[Int](tensorRank.length - 2)
      var tmpValues: Array[Double] = new Array[Double](tensorRank(0) * tensorRank(1))

      // number of matrices is the product of ranks except the first 2
      for(i <- 2 until tensorRank.length){
        unfoldRank(i - 2) = tensorRank(i)
      }
      var tmp = 0

      var unfoldMatrices: Array[BMatrix[Double]] = new Array[BMatrix[Double]](unfoldRank.product)

      for(k <- 0 to unfoldRank.product - 1; j <- 0 to (tensorRank(0) * tensorRank(1)) - 1){
        // create series of 2D Matrices
        tmp = j + k * (tensorRank(0) * tensorRank(1))
        tmpValues(j) = linearTensor.apply(tmp, 0)
        if(j == (tensorRank(0) * tensorRank(1)) - 1) {
          unfoldMatrices(k) = new BMatrix(tensorRank(0), tensorRank(1), tmpValues)
          //var listMatrices = MySpark.sc.parallelize(Seq(unfoldMatrix))
        }

      }
      MySpark.sc.parallelize(unfoldMatrices)
    }*/
  /*
    def blockTensorAsVectors(densemat: BMatrix[Double]): RDD[Vect[Double]] ={
     // val tmpRdd: RDD[Vect[Double]] = new RDD[Vect[Double]]

      val tmpRdd = matrixToRDD(densemat)

      //val fullRdd: RDD[Vect[Double]] = concatRDDs(tmpRdd, 0)
      //fullRdd
      tmpRdd
    }*/
  def blockTensorAsVectors(m: BMatrix[Double]): Array[Vect[Double]] ={
    val columns = m.toArray.grouped(m.rows)
    val rows = columns.toSeq.transpose
    val vectors = rows.map(row => row.toArray.toVector)
    vectors.toArray
  }
  /*
    def concatRDDs(r: Array[RDD[Vect[Double]]], index: Int): RDD[Vect[Double]] = {
      if(index == r.length - 1)
        r(index)
      else
        r(index) ++ concatRDDs(r, index + 1)
    }
  */
  //-----------------------------------------------------------------------------------------------------------------
  // Perform local tensor unfold
  //-----------------------------------------------------------------------------------------------------------------
  def localTensorUnfold( tensorMatrix: DenseMatrix[Double], unfoldDim: Int, tensorRank: Array[Int] )
  : DenseMatrix[Double] =
  {
    val tensorDims = tensorRank.length
    var unfoldMatrix: DenseMatrix[Double] = null
    val unfoldRank = new Array[Int](2)
    if( unfoldDim == 0 )                   // For first dimension
    {
      unfoldRank(0) = tensorRank(0)
      unfoldRank(1) = tensorRank(1)
      for( i <- 2 until tensorDims )
      {
        unfoldRank(1) = unfoldRank(1) * tensorRank(i)
      }
      //*unfoldMatrix = tensorVector.asDenseMatrix.reshape( unfoldRank(0), unfoldRank(1), false )
      unfoldMatrix = tensorMatrix.reshape( unfoldRank(0), unfoldRank(1) )
    }
    else if( unfoldDim == tensorDims - 1 ) // For last dimension
    {
      unfoldRank(0) = tensorRank(0)
      unfoldRank(1) = tensorRank(unfoldDim)
      for( i <- 1 to tensorDims - 2 )
      {
        unfoldRank(0) = unfoldRank(0) * tensorRank(i)
      }
      //*unfoldMatrix = tensorVector.asDenseMatrix.reshape( unfoldRank(0), unfoldRank(1), false ).t
      unfoldMatrix = tensorMatrix.reshape( unfoldRank(0), unfoldRank(1) ).t

    }
    else                                   // For others dimension
    {
      unfoldRank(0) = tensorRank(0)
      unfoldRank(1) = 1
      for( i <- 1 until tensorDims )
      {
        if( i < unfoldDim )
          unfoldRank(0) = unfoldRank(0) * tensorRank(i)
        else
          unfoldRank(1) = unfoldRank(1) * tensorRank(i)
      }
      val temp = tensorRank.product / tensorRank(unfoldDim)

      //*unfoldMatrix = tensorVector.asDenseMatrix.reshape( unfoldRank(0), unfoldRank(1), false ).t
      //*                           .reshape( tensorRank(unfoldDim), temp, false )
      unfoldMatrix = tensorMatrix.reshape( unfoldRank(0), unfoldRank(1) ).t
        .reshape( tensorRank(unfoldDim), temp, false )
    }

    unfoldMatrix
  }

  def localTensorRefold( tensorMatrix: DenseMatrix[Double], refoldDim: Int, tensorRank: Array[Int] )
  : DenseMatrix[Double] =
  {
    var refoldMatrix: DenseMatrix[Double] = null
    val tensorDims = tensorRank.length

    // Refold prodMatrix
    if( refoldDim == 0 )                            // For first dimension
    {
      //*refoldVector = prodMatrix.toDenseVector
      refoldMatrix = tensorMatrix.reshape( tensorMatrix.size, 1 )
    }
    else if( refoldDim == tensorDims - 1 )          // For last dimension
    {
      //*refoldVector = prodMatrix.t.toDenseVector
      refoldMatrix = tensorMatrix.t.reshape( tensorMatrix.size, 1, false )
    }
    else                                          // For other dimension
    {
      val refoldRank = new Array[Int](2)

      refoldRank(0) = tensorRank(0)
      refoldRank(1) = 1
      for( i <- 1 until tensorDims )
      {
        if( i < refoldDim )
          refoldRank(0) = refoldRank(0) * tensorRank(i)
        else
          refoldRank(1) = refoldRank(1) * tensorRank(i)
      }

      //*refoldVector = prodMatrix.reshape( refoldRank(1), refoldRank(0), false ).t.toDenseVector
      refoldMatrix = tensorMatrix.reshape( refoldRank(1), refoldRank(0) ).t.reshape( tensorMatrix.size, 1, false )
    }

    refoldMatrix
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Convert linear index to subscript index
  //-----------------------------------------------------------------------------------------------------------------
  def linear2Sub( index: Int, totalRank: Array[Int] ): CM.ArraySeq[Int] =
  {
    require( index < totalRank.product, s"Linear index out of range" )
    val dims = totalRank.length
    val subIndex = new CM.ArraySeq[Int]( dims )

    subIndex( 0 ) = index
    for( dim <- 0 until dims - 1 )
    {
      subIndex( dim + 1 ) = subIndex( dim ) / totalRank( dim )
      subIndex( dim ) = subIndex( dim ) % totalRank( dim )
    }

    subIndex
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Perform local multiple dimension matrix multiply
  //-----------------------------------------------------------------------------------------------------------------
  private def localMultiDimMMMany( blockSubIndex: CM.ArraySeq[Int], denseMatrix: DenseMatrix[Double],
                                   prodsSeq: Array[Int], bcMatrixArray: Array[Broadcast[DenseMatrix[Double]]],
                                   blockRank: Array[Int], blockNum: Array[Int], blockFinal: Array[Int],
                                   prodsTensorRank: Array[Int], prodsBlockNum: Array[Int] )
  : CM.ArraySeq[( CM.ArraySeq[Int], DenseMatrix[Double] )] =
  {
    // Count local tensor rank for input block
    val localTensorRank = blockRank.clone()
    val tensorDims = localTensorRank.length
    for( i <- 0 until tensorDims )
    {
      if( blockSubIndex(i) == blockNum(i) - 1 ) // Variable-blockNum is not zero-based
        localTensorRank(i) = blockFinal(i)
    }

    val result = new CM.ArraySeq[ (CM.ArraySeq[Int], DenseMatrix[Double]) ]( prodsBlockNum.product )

    // Perform mode-N prods
    for ( prodsBlockIndex <- 0 until prodsBlockNum.product )
    {
      var prodsMatrix = denseMatrix         // Let prodsMatrix pointer point to input block for succeeding computing
    val localProdsRank = localTensorRank.clone()

      // Get subindex of dst prods block
      val prodsBlockSubIndex = linear2Sub( prodsBlockIndex, prodsBlockNum )

      // For every dimension in prodsSeq
      for( prodDim <- prodsSeq )
      {
        // Get unfold tensor matrix( right matrix )
        val rightMatrix = localTensorUnfold( prodsMatrix, prodDim, localProdsRank )

        // Set row index of prod matrix( left matrix )
        val prodDimBlockRank = blockRank( prodDim )
        val rowStart = blockSubIndex( prodDim ) * prodDimBlockRank
        val rowIndex = rowStart to rowStart + localProdsRank( prodDim ) - 1

        // Set column index of prod matrix( left matrix )
        var colIndex = 1 to 2
        val colStart = prodsBlockSubIndex( prodDim ) * prodDimBlockRank  // Variable-iter is also zero-based
        if( prodsBlockSubIndex( prodDim ) < prodsBlockNum( prodDim ) - 1 )
          colIndex = colStart to colStart + prodDimBlockRank - 1
        else
        {
          val prodFinal = ( ( prodsTensorRank( prodDim ) - 1 ) % prodDimBlockRank ) + 1
          colIndex = colStart to colStart + prodFinal - 1
        }

        val leftMatrix = bcMatrixArray( prodDim ).value( rowIndex, colIndex )
        //*val leftMatrix = bcMatrixArray( prodDim ).value( rowIndex, colIndex ).t

        val resultMatrix = leftMatrix.t * rightMatrix
        //*val sparkRightMatrix = new SpDM( rightMatrix.rows, rightMatrix.cols, rightMatrix.toArray )
        //*val sparkLeftMatrix = new SpDM( leftMatrix.rows, leftMatrix.cols, leftMatrix.toArray )
        //*val sparkResultMatrix = sparkLeftMatrix.multiply( sparkRightMatrix )
        //*val resultMatrix = new DenseMatrix( sparkResultMatrix.numRows, sparkResultMatrix.numCols, sparkResultMatrix.toArray )

        localProdsRank( prodDim ) = resultMatrix.rows
        prodsMatrix = localTensorRefold( resultMatrix, prodDim, localProdsRank  )

      }

      result( prodsBlockIndex ) = ( prodsBlockSubIndex, prodsMatrix )
    }

    result
  }


  //-----------------------------------------------------------------------------------------------------------------
  // Perform blocks tensor unfold
  //-----------------------------------------------------------------------------------------------------------------
  def TensorUnfoldBlocks2( tensorRDDtmp: RDD[(CM.ArraySeq[Int], DMatrix)], unfoldDim: Int, tensorRank: Array[Int], blockRank: Array[Int], blockNumTmp: Array[Int])
  :RDD[(CM.ArraySeq[Int], BMatrix[Double])]
  =
  {
    var tensorRDD = tensorRDDtmp.collect()
    val tensorDims = tensorRank.length
    val unfoldRank: Array[Int] = Array(tensorRank(unfoldDim), tensorRank.product / tensorRank(unfoldDim))
    val unfoldBlockRank: Array[Int] = Array(blockRank(unfoldDim), blockRank.product / blockRank(unfoldDim))

    val unfoldSeq: Array[Int] = unfoldRank.zip(unfoldBlockRank).map{ case(x, y) => roundUp(x.toDouble / y.toDouble)}

    val totalNumBlocks = blockNumTmp.product / blockNumTmp(unfoldDim)

    var subIndex = defineSubIndex(tensorRDD.map{ case(x, y) => x}, blockNumTmp.clone(), unfoldDim)
    // permuting block numbers depending on unfolding dimension
    val blockNum = permute(unfoldDim, blockNumTmp.toList).toArray

    var newBlocks: Array[(CM.ArraySeq[Int], BMatrix[Double])] = new Array[(CM.ArraySeq[Int], BMatrix[Double])](unfoldSeq.product + unfoldSeq(0) - 1)

    for( i <- 0 until unfoldSeq(0)){
      // Valid size of the block
      var validSize = min(unfoldRank(1), unfoldBlockRank(1))
      var unfoldSize = 0
      var blockIndexCol = 1
      //println("valid size = " + validSize)
      // initialize tensor
      var unfoldBlock: BMatrix[Double] = null

      for( j <- 0 until totalNumBlocks){
        var block = tensorRDD.filter{ case (x, y) => x == subIndex(j + (i * totalNumBlocks))}.map{ case(x, y) => localTensorUnfoldBlock(y, x, unfoldDim, blockRank, blockNumTmp, tensorRank)}.take(1)
        // get the size of the block
        var blockSize = block(0).cols

        unfoldSize += blockSize


        //println(subIndex(j + (i * totalNumBlocks)) + " => block size = " + block(0).rows + " * " + blockSize)

        if(unfoldSize >= validSize){
          unfoldSize -= validSize
          var catSize = blockSize - unfoldSize
          // reshape matrix into cat size -- concatenate with previous data
          if(unfoldBlock == null)
            unfoldBlock = block(0)(::, 0 to (catSize - 1))
          else
            unfoldBlock = BMatrix.horzcat(unfoldBlock, block(0)(::, 0 to (catSize - 1)))

          // Generation of new block is complete
          // Save the new block
          // HDFS
          newBlocks(i * unfoldSeq(1) + blockIndexCol) = (CM.ArraySeq[Int](i, blockIndexCol - 1), unfoldBlock)

          // Update block index and size
          validSize = min(unfoldRank(1) - blockIndexCol * unfoldBlockRank(1), unfoldBlockRank(1))
          blockIndexCol += 1

          // Update the unfolded tensor (data left)
          unfoldBlock = block(0)(::, catSize to -1)

        } else {
          if(unfoldBlock == null)
            unfoldBlock = block(0)
          else {

            unfoldBlock = BMatrix.horzcat(unfoldBlock, block(0))
          }
        }

      }
    }
    newBlocks = newBlocks.filter{ case(x) => x != null}
   // newBlocks.map{ case(ids, mat) => ids}.foreach(println)
    MySpark.sc.parallelize(newBlocks)
    //.partitionBy(new MyPartitioner(unfoldSeq))
    //.repartition(unfoldSeq.product).reduceByKey( new MyPartitioner( unfoldSeq ), ( a, b ) => a + b )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Perform blocks tensor unfold
  //-----------------------------------------------------------------------------------------------------------------
  def TensorUnfoldBlocks( tensorRDD: RDD[(CM.ArraySeq[Int], DMatrix)], unfoldDim: Int, tensorRank: Array[Int], blockRank: Array[Int], blockNumTmp: Array[Int])
  :RDD[(CM.ArraySeq[Int], BMatrix[Double])]
  =
  {
    val tensorDims = tensorRank.length
    val unfoldRank: Array[Int] = Array(tensorRank(unfoldDim), tensorRank.product / tensorRank(unfoldDim))
    val unfoldBlockRank: Array[Int] = Array(blockRank(unfoldDim), blockRank.product / blockRank(unfoldDim))

    val unfoldSeq: Array[Int] = unfoldRank.zip(unfoldBlockRank).map{ case(x, y) => roundUp(x.toDouble / y.toDouble)}

    val totalNumBlocks = blockNumTmp.product / blockNumTmp(unfoldDim)

    var subIndex = defineSubIndex(tensorRDD.map{ case(x, y) => x}.collect(), blockNumTmp.clone(), unfoldDim)
    // permuting block numbers depending on unfolding dimension
    val blockNum = permute(unfoldDim, blockNumTmp.toList).toArray

    var newBlocks: Array[(CM.ArraySeq[Int], BMatrix[Double])] = new Array[(CM.ArraySeq[Int], BMatrix[Double])](unfoldSeq.product + unfoldSeq(0))

    for( i <- 0 until unfoldSeq(0)){
      // Valid size of the block
      var validSize = min(unfoldRank(1), unfoldBlockRank(1))
      var unfoldSize = 0
      var blockIndexCol = 1
      //println("valid size = " + validSize)
      // initialize tensor
      var unfoldBlock: BMatrix[Double] = null

      for( j <- 0 until totalNumBlocks){
        var block = tensorRDD.filter{ case (x, y) => x == subIndex(j + (i * totalNumBlocks))}.map{ case(x, y) => localTensorUnfoldBlock(y, x, unfoldDim, blockRank, blockNumTmp, tensorRank)}.take(1)
        // get the size of the block
        var blockSize = block(0).cols

        unfoldSize += blockSize


        //println(subIndex(j + (i * totalNumBlocks)) + " => block size = " + block(0).rows + " * " + blockSize)

        if(unfoldSize >= validSize){
          unfoldSize -= validSize
          var catSize = blockSize - unfoldSize
          // reshape matrix into cat size -- concatenate with previous data
          if(unfoldBlock == null)
            unfoldBlock = block(0)(::, 0 to (catSize - 1))
          else
            unfoldBlock = BMatrix.horzcat(unfoldBlock, block(0)(::, 0 to (catSize - 1)))

          // Generation of new block is complete
          // Save the new block
          // HDFS
          newBlocks(i * unfoldSeq(1) + blockIndexCol) = (CM.ArraySeq[Int](i, blockIndexCol), unfoldBlock)

          // Update block index and size
          validSize = min(unfoldRank(1) - blockIndexCol * unfoldBlockRank(1), unfoldBlockRank(1))
          blockIndexCol += 1

          // Update the unfolded tensor (data left)
          unfoldBlock = block(0)(::, catSize to -1)

        } else {
          if(unfoldBlock == null)
            unfoldBlock = block(0)
          else {

            unfoldBlock = BMatrix.horzcat(unfoldBlock, block(0))
          }
        }

      }
    }
    MySpark.sc.parallelize(newBlocks)
    //.partitionBy(new MyPartitioner(unfoldSeq))
    //.repartition(unfoldSeq.product).reduceByKey( new MyPartitioner( unfoldSeq ), ( a, b ) => a + b )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Perform local matrix multiply
  //-----------------------------------------------------------------------------------------------------------------
  def localMMMany( blockSubIndex: CM.ArraySeq[Int], denseMatrix: DenseMatrix[Double], prodDim: Int,
                           bcProdMatrix: Broadcast[DenseMatrix[Double]], blockRank: Array[Int], blockNum: Array[Int],
                           blockFinal: Array[Int] )
  : CM.ArraySeq[( CM.ArraySeq[Int], DenseMatrix[Double] )] =
  {
    // val a = denseMatrix.valueAt(1)
    val a= denseMatrix(List(1,2, 2),::)

    // Count local tensor rank
    val localTensorRank = blockRank.clone()

    val tensorDims = localTensorRank.length
    for( i <- 0 until tensorDims )
    {
      if( blockSubIndex(i) == blockNum(i) - 1 ) // Variable-blockNum is not zero-based
        localTensorRank(i) = blockFinal(i)
    }

    // This function is going to compute multiple local matrix multiplication, so it have loop to compute
    // Set row index of prod matrix( left matrix )
    val prodDimBlockRank = blockRank( prodDim )          // Variable to store block rank of prod dim
  val prodDimBlockIndex = blockSubIndex( prodDim )     // Variable to store block index of prod dim
  val rowStart = prodDimBlockIndex * prodDimBlockRank  // In this application, all index is zero-based
  val rowIndex = rowStart until rowStart + localTensorRank( prodDim )  // If block is on border,
  // variable-localTensor Rank has counted
  // remaining rank.
  // Count max iteration
  val prodDimRank = bcProdMatrix.value.cols
    val maxProdBlockIndex = ceil( prodDimRank.toFloat / prodDimBlockRank.toFloat ).toInt - 1 // For zero-based index

    // Set result collection to return
    //*var result: List[(List[Int], DenseVector[Double])] = List()
    val result = new CM.ArraySeq[ (CM.ArraySeq[Int], DenseMatrix[Double]) ]( maxProdBlockIndex + 1 )
    var colIndex = 1 to 2

    // For prodDim is not first or last dimension, compute partial refold Rank to decrease computing cost
    val refoldRank = new Array[Int](2)
    if( prodDim != 0 && prodDim != tensorDims - 1 )
    {
      refoldRank(0) = localTensorRank(0)
      refoldRank(1) = 1
      for( i <- 1 until tensorDims )
      {
        if( i < prodDim )
          refoldRank(0) = refoldRank(0) * localTensorRank(i)
        else if( i == prodDim )
          refoldRank(1) = refoldRank(1) * 1
        else
          refoldRank(1) = refoldRank(1) * localTensorRank(i)
      }
    }

    // Set unfold tensor matrix( right matrix )
    val rightMatrix = localTensorUnfold( denseMatrix, prodDim, localTensorRank )

    // Start iteration
    for( index <- 0 to maxProdBlockIndex )
    {
      // Compute column index of basis matrix and get part of basis matirx
      val colStart = index * prodDimBlockRank  // Variable-iter is also zero-based
      if( index < maxProdBlockIndex )
        colIndex = colStart to colStart + prodDimBlockRank - 1
      else
      {
        val prodFinal = ( (prodDimRank - 1) % prodDimBlockRank ) + 1
        colIndex = colStart to colStart + prodFinal - 1
      }
      val leftMatrix = bcProdMatrix.value( rowIndex, colIndex )

      // Local matrix multiplication
      var prodMatrix: DenseMatrix[Double] = null
      prodMatrix = leftMatrix.t * rightMatrix

      // Prepare null Dense Matrix to store refolded matrix
      //*var refoldVector: DenseVector[Double] = null
      var refoldMatrix: DenseMatrix[Double] = null

      // Refold prodMatrix
      if( prodDim == 0 )                            // For first dimension
      {
        //*refoldVector = prodMatrix.toDenseVector
        refoldMatrix = prodMatrix.reshape( prodMatrix.size, 1 )
      }
      else if( prodDim == tensorDims - 1 )          // For last dimension
      {
        //*refoldVector = prodMatrix.t.toDenseVector
        refoldMatrix = prodMatrix.t.reshape( prodMatrix.size, 1, false )
      }
      else                                          // For other dimension
      {
        var tempRank = 0
        if( index < maxProdBlockIndex )
        //*refoldRank(1) = refoldRank(1) * prodDimBlockRank
          tempRank = refoldRank(1) * prodDimBlockRank
        else
        {
          val prodFinal = ( (prodDimRank - 1) % prodDimBlockRank ) + 1
          //*refoldRank(1) = refoldRank(1) * prodFinal
          tempRank = refoldRank(1) * prodFinal
        }
        //*refoldVector = prodMatrix.reshape( refoldRank(1), refoldRank(0), false ).t.toDenseVector
        refoldMatrix = prodMatrix.reshape( tempRank, refoldRank(0) ).t.reshape( prodMatrix.size, 1, false )
      }

      // set proded block subindex
      //val prodBlockSubIndex = blockSubIndex.toArray
      val prodBlockSubIndex = blockSubIndex.clone()
      prodBlockSubIndex( prodDim ) = index

      // Set temp result
      //*val temp = List( ( prodBlockSubIndex.toList, refoldVector ) )
      //*result = temp ++ result
      result( index ) = ( prodBlockSubIndex, refoldMatrix )
    }

    result
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Reset ket for extract basis function
  //-----------------------------------------------------------------------------------------------------------------
  private def extractBasisResetKey( blockSubIndex: CM.ArraySeq[Int], tensorMatrix: DenseMatrix[Double], dstDim: Int )
  : ( CM.ArraySeq[Int], ( Int, DenseMatrix[Double] ) ) =
  {
    // Generate new key
    //*val key = blockSubIndex.toArray
    //*key(dstDim) = 0
    val blockIndex = blockSubIndex(dstDim)
    blockSubIndex( dstDim ) = 0

    // Output
    ( blockSubIndex, ( blockIndex, tensorMatrix ) )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Computer partial covariance matrix
  //-----------------------------------------------------------------------------------------------------------------

  private def computePartialCovMatrix( generalSubIndex: CM.ArraySeq[Int],
                                       blocks: Iterable[ (Int, DenseMatrix[Double]) ], tensorInfo: TensorInfo,
                                       dstDim: Int )
  : DenseMatrix[Double] =
  {
    // Set variable
    val blockTotal = blocks.size
    val dstDimBlockRank = tensorInfo.blockRank(dstDim)
    //*val tensorRankArray = new Array[Array[Int]](blockTotal)
    val covMatrix = DenseMatrix.zeros[Double]( tensorInfo.tensorRank(dstDim), tensorInfo.tensorRank(dstDim) )

    // Count local tensor rank for all tensor blocks
    //*for( i <- 0 until blockTotal  )
    //*{
    //*  val localTensorRank = tensorInfo.blockRank.clone()
    //*  val blockSubIndex = generalSubIndex.toArray
    //*  blockSubIndex(dstDim) = blocks.slice( i, i+1 ).head._1
    //*
    //*  for( j <- 0 until tensorInfo.tensorDims )
    //*  {
    //*    if( blockSubIndex(j) == tensorInfo.blockNum(j) - 1  )
    //*      localTensorRank(j) = tensorInfo.blockFinal(j)
    //*  }
    //*
    //*  tensorRankArray(i) = localTensorRank.clone()
    //*}

    // Count general tensor rank that is same to all blocks
    val generalTensorRank = tensorInfo.blockRank.clone()
    for( i <- 0 until tensorInfo.tensorDims )
    {
      if( generalSubIndex(i) == tensorInfo.blockNum(i) - 1  )
        generalTensorRank(i) = tensorInfo.blockFinal(i)
    }

    // Count rank of dst dimension of all block, because rank of last block is different to others
    val dstDimRankArray = new Array[Int](blockTotal)
    for( i <- 0 until blockTotal )
    {
      if( blocks.slice( i, i+1 ).head._1 != tensorInfo.blockNum(dstDim) - 1 )
        dstDimRankArray( i ) = dstDimBlockRank
      else
        dstDimRankArray( i ) = tensorInfo.blockFinal( dstDim )
    }

    // Start to compute Covariance matrix
    for( i <- 0 until blockTotal  )
    {
      val leftBlockIndex = blocks.slice( i, i+1 ).head._1
      generalTensorRank( dstDim ) = dstDimRankArray( i )
      //*val leftTensorVector = blocks.slice( i, i+1 ).head._2
      val leftTensorMatrix = localTensorUnfold( blocks.slice( i, i+1 ).head._2, dstDim, generalTensorRank )
      val rowStart = leftBlockIndex * dstDimBlockRank
      val rowIndex = rowStart to rowStart + dstDimRankArray(i) - 1

      for( j <- i until blockTotal  )
      {
        val rightBlockIndex = blocks.slice( j, j+1 ).head._1
        generalTensorRank( dstDim ) = dstDimRankArray( j )
        //*val rightTensorVector = blocks.slice( j, j+1 ).head._2
        val rightTensorMatrix = localTensorUnfold( blocks.slice( j, j+1 ).head._2, dstDim, generalTensorRank )
        val colStart = rightBlockIndex * dstDimBlockRank
        val colIndex = colStart to colStart + dstDimRankArray(j) - 1

        //tempMatrix( rowIndex, colIndex ) :=
        //  tempMatrix( rowIndex, colIndex ) + leftTensorMatrix * rightTensorMatrix.t
        val tempMatrix = leftTensorMatrix * rightTensorMatrix.t
        covMatrix( rowIndex, colIndex ) := tempMatrix
        if ( i != j )
          covMatrix( colIndex, rowIndex ) := tempMatrix.t
      }
    }

    covMatrix
  }

  def matrixToRDD(m: BMatrix[Double]): Array[Vect[Double]]  = {
    val columns = m.toArray.grouped(m.rows)
    val rows = columns.toSeq.transpose

    val vectors = rows.map(row => row.toArray.toVector)
    vectors.toArray
  }

  /*def TestmatrixToRDD(m: DMatrix, columns: Int, rows: Int): RDD[MVector] = {
    val columns = m.toArray.grouped(m.numRows)
    val rows = columns.toSeq.transpose
    val vectors = rows.map(row => Vectors.dense(row.toArray))
    MySpark.sc.parallelize(vectors)
  }*/

  def linearSub(index: Int, totalRank: Array[Int]): CM.ArraySeq[Int] = {
    require(index < totalRank.product, s"Linear index out of range")
    val dims = totalRank.length
    val subIndex = new CM.ArraySeq[Int]( dims )

    subIndex(0) = index
    for(dim <- 0 until dims - 1)
    {
      subIndex(dim + 1) = subIndex(dim) / totalRank(dim)
      subIndex(dim) = subIndex(dim) % totalRank(dim)
    }
    subIndex
  }
  /*
    // returns array of ID
    def tensor2Vects(v: RDD[MVector], tensorRanks: Array[Int]): RDD[MVector] = {
      val tensor = null
      for(dim <- 0 until tensorRanks.length - 1){
        for(nb <-0 until tensorRanks(dim) * tensorRanks(dim + 1) by tensorRanks(dim)
        val ids = Vector(null)


      }
    }*/
}