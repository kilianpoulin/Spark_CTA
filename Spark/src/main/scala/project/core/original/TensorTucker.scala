package project.core.original

/**
  * Created by root on 2015/12/21.
  */
import breeze.linalg._
import breeze.numerics.abs
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.collection.{mutable => CM}
import scala.util.control.Breaks

object TensorTucker
{
  var sc: SparkContext = null

  def setSparkContext( inSC: SparkContext ): Unit =
  {
    sc = inSC
//    Tensor.setSparkContext( sc )
  }

  def deComp ( tensorRDD: RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ], tensorInfo: Tensor.TensorInfo,
               deCompSeq: Array[Int], coreRank: Array[Int], maxIter: Int, epsilon: Double )
  : ( RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ], Tensor.TensorInfo, Array[ Broadcast[DenseMatrix[Double]] ], Int )
  =
  {
    // Create random basis matrices and broadcast them
    val deCompDims = deCompSeq.length
    val bcBasisMatrixArray = new Array[ Broadcast[DenseMatrix[Double]] ]( tensorInfo.tensorDims )
    for ( i <- 0 until deCompDims )
    {
      //*bcBasisMatrixArray(i) = sc.broadcast( DenseMatrix.rand( tensorInfo.tensorRank( deCompSeq(i) ), coreRank(i) ) )
      bcBasisMatrixArray( deCompSeq(i) ) =
        sc.broadcast( DenseMatrix.rand( tensorInfo.tensorRank( deCompSeq(i) ), coreRank( deCompSeq(i) ) ) )
    }

    // Set variable to record final result
    var iterRecord = 0
    var finalRDD: RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ] = null
    var finalInfo: Tensor.TensorInfo = null

    // Main iteration to compute tensor decomposition
    val loop = new Breaks
    loop.breakable
    {
      for( iter <- 1 to maxIter )
      {
        var convergeFlag = true

        // For every decompose dimensions to compute basis matrix
        //*for( dimCount <- 0 to deCompDims - 1 )
        for( dimCount <- deCompSeq )
        {
          // Set variable
          val prodsSeq = new Array[Int]( deCompDims - 1 )

          var cumIndex = 0
          for( i <- 0 until deCompDims )
          {
            if( deCompSeq( i ) != dimCount )
            {
              prodsSeq( cumIndex ) = deCompSeq( i )
              //*prodsRank( cumIndex ) = coreRank( i )
              //*prodsBasis( cumIndex ) = bcBasisMatrixArray( i )
              cumIndex = cumIndex + 1
            }
          }

          // Mode-N products
          val( prodsRDD, prodsInfo ) = Tensor.modeNProducts( tensorRDD, tensorInfo, prodsSeq, bcBasisMatrixArray )

          // Extract basis
          val basisMatrix = Tensor.extractBasis( prodsRDD, prodsInfo, dimCount, coreRank( dimCount ) )

          // Check converge or not
          val preBasisMatrix = bcBasisMatrixArray( dimCount ).value
          convergeFlag = convergeFlag && checkBasisMatrixConverge( preBasisMatrix, basisMatrix, epsilon )

          // Broadcast new basis matrix
          bcBasisMatrixArray( dimCount ).destroy()
          bcBasisMatrixArray( dimCount ) = sc.broadcast( basisMatrix )

          //
          /*
          if( convergeFlag == true )
          {
            finalRDD = prodsRDD
            finalInfo = prodsInfo
          }
          */
          finalRDD = prodsRDD
          finalInfo = prodsInfo
        }

        // If converge, break main iteration
        if( convergeFlag == true )
        {
          iterRecord = iter
          loop.break
        }
        else
          finalRDD.unpersist( false )
      }
    }

    // Post processing
    val ( coreRDD, coreInfo ) = Tensor.modeNProduct( finalRDD, finalInfo, deCompSeq.last,
      bcBasisMatrixArray( deCompSeq.last ) )

    ( coreRDD, coreInfo, bcBasisMatrixArray, iterRecord )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Reconst
  //-----------------------------------------------------------------------------------------------------------------
  def reConst ( coreRDD: RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ], coreInfo: Tensor.TensorInfo,
                reContSeq: Array[Int], bcBasisMatrixArray: Array[ Broadcast[DenseMatrix[Double]] ] )
  : ( RDD[ (CM.ArraySeq[Int], DenseMatrix[Double]) ], Tensor.TensorInfo ) =
  {
    val tensorDims = coreInfo.tensorDims
    val bcBasisMatrixTransArray = new Array[ Broadcast[DenseMatrix[Double]] ]( tensorDims )
    for ( i <- 0 to tensorDims - 1 )
    {
      if( bcBasisMatrixArray(i) != null )
      {
        val tempMatrix = bcBasisMatrixArray( i ).value.t
        bcBasisMatrixTransArray( i ) = sc.broadcast( tempMatrix )
      }
    }

    val( reconstRDD, reconstInfo ) = Tensor.modeNProducts( coreRDD, coreInfo, reContSeq, bcBasisMatrixTransArray )
    val temp = reconstRDD.count()

    ( reconstRDD, reconstInfo )
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Check previous basis matrix and current basis matrix is converge or not
  //-----------------------------------------------------------------------------------------------------------------
  def checkBasisMatrixConverge( matrixA: DenseMatrix[Double], matrixB: DenseMatrix[Double], threshold: Double )
  : Boolean =
  {
    // Set variable
    val columnS = matrixA.cols
    val onesVector = DenseVector.ones[Double]( columnS )
    val thresholdVector = DenseVector.fill( columnS ){ threshold }

    // Compute
    //val flag = abs( onesVector - abs( sum( matrixA :* matrixB, Axis._0 ).toDenseVector ) ) :< thresholdVector
    //flag.forall( _ == true )
    true
  }
}