package project.core.v2

/**
  * Created by root on 2015/12/21.
  */
import breeze.linalg._
import breeze.linalg.operators.OpPow
import breeze.numerics.abs
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{DenseMatrix => DMatrix}
import project.core.v2
import project.core.v2.Tensor.TensorInfo

import scala.annotation.tailrec
import scala.collection.{mutable => CM}
import scala.util.control.Breaks

object TensorTucker {
  var sc: SparkContext = null

  def setSparkContext(inSC: SparkContext): Unit = {
    sc = inSC
    //    Tensor.setSparkContext( sc )
  }

  implicit def bool2int(b: Boolean) = if (b) 1 else 0

  def h(index: Int, blkrank: Int, tensrank: Int, num: Int) = if (index == num) (tensrank - blkrank) % blkrank else blkrank

  def computeApprox(tensorRDD: RDD[(CM.ArraySeq[Int], DenseMatrix[Double])], tensorInfo: Tensor.TensorInfo, cluNum: Int, coreTensors: RDD[(CM.ArraySeq[Int], DenseMatrix[Double])],
                    coreInfo: Tensor.TensorInfo, basisMatrices: Array[Broadcast[DenseMatrix[Double]]], deCompSeq: Array[Int], cluDim: Int): DenseVector[Double] = {
    // unfolding the block-wise core tensors
    var unfoldedCoreTensor: DenseMatrix[Double] = null

    var newCoreInfo = new v2.Tensor.TensorInfo(coreInfo.tensorDims, Array(50,50,20,20), tensorInfo.blockRank, Array(1,1,2,2), Array(50,50,6,6))

    var coreTmp = Tensor.TensorUnfoldBlocks( coreTensors.map{ case(x,y) => (x, new (org.apache.spark.mllib.linalg.DenseMatrix)(y.rows, y.cols, y.valuesIterator.toArray))}, cluDim, newCoreInfo.tensorRank, newCoreInfo.blockRank, newCoreInfo.blockNum).filter{ x => x != null}
    // assemble
    var newmat = coreTmp.map{ case(x,y) => y}.reduce{(a,b) => DenseMatrix.horzcat(a,b)}

    var squaremat = newmat * newmat.t

    // dual moden basis matrix
    //val eigPair = eigSym(squaremat)
    val svd.SVD(u,s,v) = svd(squaremat)
    val dualbasis = v

    // project eachs slice on the dual mode basis matrix
    val prodsSeq = deCompSeq.filter{ x => x != cluDim}
    val(approx, approxInfo) = Tensor.modeNProducts(tensorRDD, tensorInfo, prodsSeq, basisMatrices)
    // unfolding the result
    val unfApprox = Tensor.TensorUnfoldBlocks(approx.map{ case(x,y) =>
      (x, new (org.apache.spark.mllib.linalg.DenseMatrix)(y.rows, y.cols, y.valuesIterator.toArray))}, cluDim, approxInfo.tensorRank, approxInfo.blockRank, approxInfo.blockNum)
      .filter{ x => x != null}

    val squareApprox = unfApprox.map{ case(x,y) => y * y.t}
    val tmpApprox = squareApprox.map{ x => calcNorm(x * reshapeBasis(dualbasis, x.rows))}
    val finalApprox = assembleBlock(tmpApprox, tensorInfo.blockRank(cluDim), tensorInfo.blockNum(cluDim), tensorInfo.blockFinal(cluDim))
    //val finalApprox = squareApprox.map{ x => calcNorm(x * reshapeBasis(dualbasis, x.rows))}
    finalApprox
  }

  def assembleBlock(blocksRDD: RDD[DenseVector[Double]], blockRank: Int, blockNum: Int, blockFinal: Int): DenseVector[Double] ={
    var length = (blockNum - 1) * blockRank
    var block1 = blocksRDD.filter{ x => x.length == length}.reduce((a,b) => a + b)
    var block2 = blocksRDD.filter{ x => x.length == blockFinal}.reduce((a,b) => a + b)
    DenseVector.vertcat(block1, block2)
  }

  def reshapeBasis(basis: DenseMatrix[Double], size: Int): DenseMatrix[Double] ={
    var newBasis: DenseMatrix[Double] = DenseMatrix.zeros[Double](size, size)

    if(basis.rows != size){
      val seq = (size to basis.rows - 1)
      newBasis = basis.delete(seq, Axis._0).delete(seq, Axis._1)
    }
    newBasis
  }

  def calcNorm(matrix: DenseMatrix[Double]): DenseVector[Double] ={
    var normMatrix: DenseVector[Double] = DenseVector.zeros[Double](matrix.cols)
    for(i <- 0 to matrix.cols - 1){
      normMatrix(i) = norm(matrix(::,i))
    }
    normMatrix
  }


/*
  def calcNorm(matrix: DenseMatrix[Double], i: Int): Double ={
   if(i == matrix.rows - 1)
     norm(matrix(::, i))
   else
     norm(matrix(::, i)) + calcNorm(matrix, i + 1)
  }*/

  def deComp2(tensorRDD: RDD[(CM.ArraySeq[Int], DenseMatrix[Double])], tensorInfo: Tensor.TensorInfo,
              deCompSeq: Array[Int], coreRank: Array[Int], maxIter: Int, epsilon: Double)
  : (RDD[(CM.ArraySeq[Int], DenseMatrix[Double])], Tensor.TensorInfo, Array[Broadcast[DenseMatrix[Double]]], Int)
  = {

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

    /*
    // Create random basis matrices and broadcast them
    val deCompDims = deCompSeq.length
    val bcBasisMatrixArray = new Array[Broadcast[DenseMatrix[Double]]](tensorInfo.tensorDims)
    for (i <- 0 until deCompDims) {
      bcBasisMatrixArray(deCompSeq(i)) =
        sc.broadcast(DenseMatrix.rand(tensorInfo.tensorRank(deCompSeq(i)), coreRank(deCompSeq(i))))
    }

    // Initialize basis matrices
    // Set variable to record final result
    var iterRecord = 0
    var finalRDD: RDD[(CM.ArraySeq[Int], DenseMatrix[Double])] = null
    var finalInfo: Tensor.TensorInfo = null


    finalRDD = tensorRDD
    finalInfo = tensorInfo

    var firstdim: RDD[(CM.ArraySeq[Int], DenseMatrix[Double])] = null
    var firstdimInfo: Tensor.TensorInfo = null
    // For every decompose dimensions to compute basis matrix
    for (dimCount <- deCompSeq) {

      if(dimCount != deCompSeq.last) {
        // Extract basis
        val (basisMatrix, eigenVal) = Tensor.extractBasis(finalRDD, finalInfo, dimCount, coreRank(dimCount))

        // Mode-N products
        val (prodsRDD, prodsInfo) = Tensor.modeNProduct(finalRDD, finalInfo, dimCount, MySpark.sc.broadcast(basisMatrix))

        if(dimCount == 1){
         firstdim = prodsRDD
          firstdimInfo = prodsInfo
        }
        // Broadcast new basis matrix
        bcBasisMatrixArray(dimCount).destroy()
        bcBasisMatrixArray(dimCount) = sc.broadcast(basisMatrix)

        finalRDD = prodsRDD
        finalInfo = prodsInfo
      } else {

        val (basisMatrix, eigenVal) = Tensor.extractBasis(finalRDD, finalInfo, dimCount, coreRank(dimCount))

        finalRDD = firstdim
        finalInfo = firstdimInfo
        // Broadcast new basis matrix
        bcBasisMatrixArray(dimCount).destroy()
        bcBasisMatrixArray(dimCount) = sc.broadcast(basisMatrix)
      }

  }


  // Main iteration to compute tensor decomposition
  val loop = new Breaks
  loop.breakable {
    for (iter <- 1 to maxIter) {
      var convergeFlag = true

      // For every decompose dimensions to compute basis matrix
      for (dimCount <- deCompSeq) {

        val prodsSeq = new Array[Int](deCompDims - 1)

        var cumIndex = 0
        for (i <- 0 until deCompDims) {
          if (deCompSeq(i) != dimCount) {
            prodsSeq(cumIndex) = deCompSeq(i)
            cumIndex = cumIndex + 1
          }
        }
        // Mode-N products
        val (prodsRDD, prodsInfo) = Tensor.modeNProducts(finalRDD, finalInfo, prodsSeq, bcBasisMatrixArray)

        val (basisMatrix, eigenVal) = Tensor.extractBasis(prodsRDD, prodsInfo, dimCount, coreRank(dimCount))

        // Check converge or not
        val preBasisMatrix = bcBasisMatrixArray(dimCount).value
        convergeFlag = convergeFlag && checkBasisMatrixConverge(preBasisMatrix, basisMatrix, epsilon)


        // If converge, break main iteration
        if (convergeFlag == true) {
          iterRecord = iter
          loop.break
        }
        else
          finalRDD.unpersist(false)

      }
    }
  }
    // Post processing
    val (coreRDD, coreInfo) = Tensor.modeNProduct(finalRDD, finalInfo, deCompSeq.last,
      bcBasisMatrixArray(deCompSeq.last))

    (coreRDD, coreInfo, bcBasisMatrixArray, iterRecord)
  }*/

  //-----------------------------------------------------------------------------------------------------------------
  // Reconst
  //-----------------------------------------------------------------------------------------------------------------
  def reConst(coreRDD: RDD[(CM.ArraySeq[Int], DenseMatrix[Double])], coreInfo: Tensor.TensorInfo,
              reContSeq: Array[Int], bcBasisMatrixArray: Array[Broadcast[DenseMatrix[Double]]])
  : (RDD[(CM.ArraySeq[Int], DenseMatrix[Double])], Tensor.TensorInfo) = {
    val tensorDims = coreInfo.tensorDims
    val bcBasisMatrixTransArray = new Array[Broadcast[DenseMatrix[Double]]](tensorDims)
    for (i <- 0 to tensorDims - 1) {
      if (bcBasisMatrixArray(i) != null) {
        val tempMatrix = bcBasisMatrixArray(i).value.t
        bcBasisMatrixTransArray(i) = sc.broadcast(tempMatrix)
      }
    }

    val (reconstRDD, reconstInfo) = Tensor.modeNProducts(coreRDD, coreInfo, reContSeq, bcBasisMatrixTransArray)
    val temp = reconstRDD.count()

    (reconstRDD, reconstInfo)
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Reconst
  //-----------------------------------------------------------------------------------------------------------------
  def reConst2(tensorRDD: RDD[(CM.ArraySeq[Int], DenseMatrix[Double])], cluIDs: Array[Int], coreRDD: RDD[(CM.ArraySeq[Int], DenseMatrix[Double])], coreInfo: Tensor.TensorInfo,
              reContSeq: Array[Int], bcBasisMatrixArray: Array[Broadcast[DenseMatrix[Double]]])
  : (RDD[(CM.ArraySeq[Int], DenseMatrix[Double])], Tensor.TensorInfo) = {
    val tensorDims = coreInfo.tensorDims
    val bcBasisMatrixTransArray = new Array[Broadcast[DenseMatrix[Double]]](tensorDims)
    for (i <- 0 to tensorDims - 1) {
      if (bcBasisMatrixArray(i) != null) {
        val tempMatrix = bcBasisMatrixArray(i).value.t
        bcBasisMatrixTransArray(i) = sc.broadcast(tempMatrix)
      }
    }
    var (reconstRDD, reconstInfo) = Tensor.modeNProducts(coreRDD, coreInfo, reContSeq, bcBasisMatrixTransArray)


    reconstRDD = reconstRDD.map{ case(x,y) => (x, Tensor.reshapeBlock(x, 0, reconstInfo, y))}


    (reconstRDD, reconstInfo)
  }

  //-----------------------------------------------------------------------------------------------------------------
  // Check previous basis matrix and current basis matrix is converge or not
  //-----------------------------------------------------------------------------------------------------------------
  def checkBasisMatrixConverge(matrixA: DenseMatrix[Double], matrixB: DenseMatrix[Double], threshold: Double)
  : Boolean = {
    // Set variable
    val columnS = matrixA.cols
    val onesVector = DenseVector.ones[Double](columnS)
    val thresholdVector = DenseVector.fill(columnS) {
      threshold
    }

    // Compute
    val flag = abs( onesVector - abs( sum( matrixA :* matrixB, Axis._0 )).inner ) :> thresholdVector
    //val flag = abs( onesVector - abs( sum( matrixA :* matrixB, Axis._0 )) )
    //<: thresholdVector
    flag.forall( _ == false )
    //true
  }


  def transformClusters(tensorRDD: RDD[(CM.ArraySeq[Int], DMatrix)], clusterRanks: Array[Int], clusterMembers: Array[Int],
                        tensorInfo: Tensor.TensorInfo, coreRank: Array[Int], cluNum: Int, deCompDim: Int):
  (RDD[Array[(CM.ArraySeq[Int], DenseMatrix[Double])]], Array[Tensor.TensorInfo]) = {
    var outBlockMod = tensorInfo.tensorRank.map {
      _ + 1
    }.zipWithIndex.map { case (x, nb) => x % tensorInfo.blockRank(nb) }.map {
      _ - 1
    }

    var outRank = tensorInfo.tensorRank.clone()
    var outBlockNum = tensorInfo.blockNum.clone()

    var rankCell: Array[Array[Int]] = Array.ofDim(tensorInfo.tensorRank.length, tensorInfo.tensorRank.length)
    var rankNum: Int = 0
    var eleNum: Int = 0

    var eleCell: Array[List[List[Int]]] = Array.ofDim(tensorInfo.tensorRank.length)
    var validCell: Array[Array[Array[Array[Boolean]]]] = Array.ofDim(tensorInfo.tensorRank.length)
    var outValidCell: Array[Array[Array[Boolean]]] = Array.ofDim(tensorInfo.tensorRank.length)
    var blockCell: Array[Array[Int]] = Array.ofDim(tensorInfo.tensorRank.length)

    var blockCelltmp: Array[Array[Int]] = Array.ofDim(tensorInfo.tensorRank.length)
    var subIndex: CM.ArraySeq[Int] = new CM.ArraySeq[Int](tensorInfo.tensorRank.length)
    var blkRank: Array[Int] = Array.ofDim(tensorInfo.tensorRank.length)
    var validEleCell: Array[Array[Array[Boolean]]] = Array.ofDim(tensorInfo.tensorRank.length)
    var outBlockCell: Array[Array[Int]] = Array.ofDim(tensorInfo.tensorRank.length)
    var outSubIndex: CM.ArraySeq[Int] = new CM.ArraySeq[Int](tensorInfo.tensorRank.length)
    var outIndexCell: Array[Array[Int]] = Array.ofDim(tensorInfo.tensorRank.length)
    var indexCell: Array[Array[Int]] = Array.ofDim(tensorInfo.tensorRank.length)
    var clustersInfo: Array[Tensor.TensorInfo] = Array.ofDim(cluNum)

    var newclusters: Array[Array[(CM.ArraySeq[Int], DenseMatrix[Double])]] = Array.ofDim(cluNum, tensorRDD.count().toInt)


    var clusterIDs = clusterMembers.zipWithIndex.groupBy(_._1).mapValues(_.map(_._2)).map { x => x._2 }.toArray

    /**
      * Each cluster, starting from the last
      * outRank => cluster rank => number of rows in the cluster
      * eleCell => IDs of vectors for each dimension, based on the result of the KMeans clustering / block-wise version
      *
      */
    for (i <- cluNum to 1 by -1) {
      // rank of the cluster (number of rows)

      outRank(deCompDim) = clusterRanks(i - 1)

      // for each block of each dimension -- initialize vector IDs
      for (d <- 0 to (tensorInfo.tensorDims - 1)) {
        var rangevals = (0 to (outRank(d) - 1) by tensorInfo.blockRank(d)).toArray ++ Array(outRank(d))
        rangevals = rangevals.filter { x => x != 0 }
        var lists: Array[List[Int]] = Array.ofDim(rangevals.length)
        var count = 0
        for (rangeval <- rangevals) {
          lists(count) = ((1 + count * tensorInfo.blockRank(d)) to rangeval).toList
          //lists(count) = ((1 + count * tensorInfo.blockRank(d)) to ((rangeval + count) % (tensorInfo.blockRank(d) + 1))).toList
          count = count + 1
        }
        eleCell(d) = lists.toList
        outBlockMod( d ) = ( (outRank( d ) - 1) % tensorInfo.blockRank( d ) ) + 1
        outBlockNum( d ) = (outRank( d ) - outBlockMod( d )) / tensorInfo.blockRank( d ) + 1
      }

      // on peut remplir les valeurs de la dimension decomposee avec KMeans et pour le cluster etudie
      eleCell(deCompDim) = List(clusterIDs.toArray.apply(i - 1).map { x => x + 1 }.toList)


      // For each dimension
      for (d <- 0 to (tensorInfo.tensorDims - 1)) {
        // get the minimum rank of each block (starts by, then rank of 1 block, etc..)
        rankCell(d) = (0 to (tensorInfo.tensorRank(d) - 1) by tensorInfo.blockRank(d)).toArray ++ Array(tensorInfo.tensorRank(d))

        rankNum = rankCell(d).length - 1
        eleNum = eleCell(d).length
        validCell(d) = Array.ofDim(rankNum)
        outValidCell(d) = Array.ofDim(eleNum, rankNum)

        for (r <- 0 to rankNum - 1) {
          validCell(d)(r) = new Array(eleNum)
          for (e <- 0 to eleNum - 1) {
            // if the vector ID is greater than the minimum rank and lower than the maximum rank (min rank of next block)
            // then the vector #x = true => vector id is in this block
            // otherwise => false
            validCell(d)(r)(e) = eleCell(d)(e).map { x => ((x > rankCell(d)(r)) && x <= rankCell(d)(r + 1)) }.toArray

            // if there are some vector IDs true (meaning they will be added to this cluster)
            // => then outvalidcell = true
            outValidCell(d)(e)(r) = validCell(d)(r)(e).contains(true)
          }
        }

        blockCelltmp(d) = outValidCell(d).map { e => e.zipWithIndex.filter { case (r, id) => r == true }.map { case (r, id) => id } }.flatMap(x => x).toArray

      }

      var values = 1
      for(l <- blockCelltmp.length - 1 to 0 by - 1 ){
        values *= blockCelltmp(l).length
        var tmpvalues: List[Int] = List()
        for(r <- 0 to blockCelltmp(l).length - 1) {
          if(r == 0)
            tmpvalues = List.fill(tensorRDD.count().toInt / values)(blockCelltmp(l)(r))
          else
            tmpvalues = tmpvalues ++ List.fill(tensorRDD.count().toInt / values)(blockCelltmp(l)(r))
        }

        blockCell(l) = (List.fill(tensorRDD.count().toInt / tmpvalues.length)(tmpvalues)).flatMap{ x => x }.toArray
      }

      /**
        * ***************************************** END OF INITIALIZATION ************************************
        * EACH BLOCK
        */


      // read each block
      for (u <- 0 to blockCell(0).length - 1) {
        // for each dimension
        for (d <- 0 to (tensorInfo.tensorDims - 1)) {
          subIndex(d) = blockCell(d)(u)
          blkRank(d) = rankCell(d)(subIndex(d))

          // extract flags of valid elements in current block for current dimension
          validEleCell(d) = validCell(d)(subIndex(d))

          // Obtain valid output block indices for current dimension
          outBlockCell(d) = outValidCell(d).map { x => x(subIndex(d)) }.zipWithIndex.filter { case (value, id) => value == true }.map { case (value, id) => id }

        }
        // get initial tensor block
        var tensortest = tensorRDD.collect()
        var block = tensorRDD.filter { case (id, value) => id == subIndex }.collect()
        var blockID = subIndex

        // Obtain valid elements from each sub-block of the output tensor
        for (o <- 0 to outBlockCell(0).length - 1) {
          for (d <- 0 to tensorInfo.tensorDims - 1) {
            // Extract sub-indices of current output block for current dimension
            outSubIndex(d) = outBlockCell(d)(o)

            // Obtain the indices of the specified elements in current input and output blocks for current dimension
            outIndexCell(d) = validEleCell(d)(outSubIndex(d)).map { x => x: Int }
            indexCell(d) = (eleCell(d)(outSubIndex(d)).toArray zip outIndexCell(d)).filter { case (value, flag) => flag == 1 }.map { case (value, flag) => (value - 1 - blkRank(d)) }

          }

          // converting the tensor block matrix to a breeze densematrix
          var mat = block.map { case (x, y) => y }
          var bmatblock: DenseMatrix[Double] = new DenseMatrix(mat(0).numRows, mat(0).numCols, mat(0).toArray)

          // looking for the rank of the current tensor block
          var currBlockIndices = subIndex.map { x => x + 1 }.zipWithIndex
            .map { case (value, id) => h(value, tensorInfo.blockRank(id), tensorInfo.tensorRank(id), tensorInfo.blockNum(id)) }

          //saving the decomposition dimension block rank
          var decompBlk = currBlockIndices(deCompDim)
          // removing the rank the decomposition dimension
          currBlockIndices = currBlockIndices.zipWithIndex.filter { case (x, y) => y != deCompDim }.map { case (x, y) => x }

          // reshaping the tensor block matrix to get as many rows as the block rank of the decomposition dimension
          var reshapedblock = bmatblock.reshape(decompBlk, currBlockIndices.product)

          // row data IDs that will not be used in this cluster
          var newIndex2 = (0 to decompBlk - 1 by 1).filter { x => indexCell(deCompDim).contains(x) == false }

          var newmat = DenseMatrix.zeros[Double](outRank(deCompDim), currBlockIndices.product)

          if(u != 0 && (newclusters.toArray).apply(i - 1).apply(u - 1)._1 == outSubIndex){

            var tmp = ((newclusters.toArray).apply(i - 1).apply(u - 1)._2).copy
            newmat = tmp.reshape((outRank.toArray).apply(deCompDim), (currBlockIndices.toArray).product)
            for(g <- newmat.rows - 1 to (newmat.rows - indexCell(deCompDim).length) by -1){
              newmat(g, ::) := reshapedblock(indexCell(deCompDim)(newmat.rows - (g + 1)), ::)
            }

          } else {
            for(g <- 0 to indexCell(deCompDim).length - 1){
              //val row: DenseVector[Double] =
              newmat(g, ::) := reshapedblock(indexCell(deCompDim)(g), ::)
            }
          }

          // reshape the matrix into only one row
          var lastmat = newmat.reshape(1, newmat.rows * newmat.cols)

          // saving this matrix in the corresponding block cluster
          newclusters(i - 1)(u) = (outSubIndex.clone(), lastmat)

        }

      }
      // delete unecessary subtensors
      newclusters(i - 1) = newclusters(i - 1).zipWithIndex.filter{ x => x._2 % 2 != 0}.map{ x => x._1}

      var blockRanktmp = tensorInfo.blockRank.clone()
      if(outRank(deCompDim) < blockRanktmp(deCompDim))
        blockRanktmp(deCompDim) = outRank(deCompDim)
      clustersInfo(i - 1) = new Tensor.TensorInfo(tensorInfo.tensorDims, outRank.clone(), blockRanktmp, outBlockNum.clone(), outBlockMod.clone())

    }


    (MySpark.sc.parallelize(newclusters), clustersInfo)
  }

}