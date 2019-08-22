package project.core.officialApprox;
/*
public class test
//******************************************************************************************************************
// Initialize variables
//******************************************************************************************************************
  /*var clusters: RDD[Array[(CM.ArraySeq[Int], DenseMatrix[Double])]] = MySpark.sc.parallelize(Seq(null))
  var clustersInfo: Array[Tensor.TensorInfo] = Array.fill(cluNum)(null)
  var coreTensors: Array[RDD[(CM.ArraySeq[Int], DenseMatrix[Double])]] = Array.fill(cluNum)(MySpark.sc.parallelize(Seq(null)))
  var coreInfo: Array[Tensor.TensorInfo] = Array.fill(cluNum)(null)
  var basisMatrices: Array[Array[Broadcast[DenseMatrix[Double]]]] = Array.fill(cluNum)(null)
  val deCompSeq2: Array[Int] = Array(1, 2, 3)
  var reconstTensor: RDD[(CM.ArraySeq[Int], DenseMatrix[Double])] = MySpark.sc.parallelize(Seq(null))


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

  if (clustersPath == "") {
    //******************************************************************************************************************
    // Section to perform K-Means
    //******************************************************************************************************************

    val kmeans = new KMeans.KMeansClustering(cluNum, centroidsInit, centroidsPath, maxIter, tensorInfo.tensorDims, tensorInfo)
    val (clusteredRDD, clusterMembers) = kmeans.train(tensorBlocks)


    //************************************************** ****************************************************************
    // Section to perform initial tensor decomposition
    //******************************************************************************************************************

    // convert Densevectors to DenseMatrix
    var clusteredRDDmat = clusteredRDD.map { x => (x.size, x(0).length, x.map { y => y.toArray }) }.collect()
    var clusteredRDDmat2 = clusteredRDDmat.map { x => new DenseMatrix(x._2, x._1, x._3.reduce((a, b) => a ++ b)) }.map { x => (CM.ArraySeq[Int](0, 0), x.t) }


    // use vector IDs to create clusters
    val tuple = TensorTucker.transformClusters(tensorRDD, clusteredRDDmat2.map { case (x, y) => y.rows }, clusterMembers,
      tensorInfo, coreRank.clone(), cluNum, 0)

    clusters = tuple._1
    clustersInfo = tuple._2

    for (c <- 0 to cluNum - 1) {
      val clInfo = clustersInfo(c)

      val cluster = clusters.zipWithIndex.filter { case (x, y) => y == c }.map { case (x, y) => x }.flatMap { x => x }
        .reduceByKey(new MyPartitioner(clInfo.blockNum), (a, b) => a + b)

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
        .map { s => s.map { x => (x._1, new DenseMatrix[Double](x._2.numRows, x._2.numCols, x._2.values)) } }
      if (c == 0)
        clusters = oneCl
      else
        clusters = clusters.union(oneCl)
    }
  }

  if (basisPath == "" || corePath == "") {

    for (c <- 0 to cluNum - 1) {
      val tuple = TensorTucker.deComp2(clusters.zipWithIndex.filter { case (x, y) => y == c }.map { case (x, y) => x }.flatMap { x => x }, clustersInfo(c), deCompSeq, coreRank, maxIter, epsilon)
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

    for (c <- 0 to cluNum - 1) {

      // start with previously calculated coreTensor and basisMatrices
      basisMatrices(c) = Tensor.readBasisMatrices(basisPath + c + "/", deCompSeq).map { x => MySpark.sc.broadcast(x) }
      coreInfo(c) = Tensor.readTensorHeader(corePath + c + "/")
      coreTensors(c) = Tensor.readTensorBlock(corePath + c + "/", coreInfo(c).blockRank).map { x => (x._1, new DenseMatrix[Double](x._2.numRows, x._2.numCols, x._2.values)) }
    }

  }

  //******************************************************************************************************************
  // Section to perform tensor approximation
  //******************************************************************************************************************

  val approxErr: DenseMatrix[Double] = DenseMatrix.zeros[Double](tensorInfo.tensorRank(unfoldDim), cluNum)

  if (finalPath == "") {
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

        val tuple = TensorTucker.deComp2(clusters.zipWithIndex.filter { case (x, y) => y == c }.map { case (x, y) => x }.flatMap { x => x }, clustersInfo(c), deCompSeq, coreRank, maxIter, epsilon)
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


  // Shut down Spark context
  MySpark.sc.stop
}

  //******************************************************************************************************************
  // Section to do tensor Tucker reconstruction
  //******************************************************************************************************************
/*
    // Reconstruct input tensor
    if( reconstFlag == 1 )
    {

      if(finalPath != ""){

        for(c <- 0 to cluNum - 1){

          // start with previously calculated coreTensor and basisMatrices
          basisMatrices(c) = Tensor.readBasisMatrices(basisPath + finalPath + c + "/", deCompSeq).map{ x => MySpark.sc.broadcast(x)}
          coreInfo(c) = Tensor.readTensorHeader(corePath + finalPath + c + "/")
          coreTensors(c) = Tensor.readTensorBlock(corePath + finalPath + c + "/", coreInfo(c).blockRank).map{ x => (x._1, new DenseMatrix[Double](x._2.numRows, x._2.numCols, x._2.values))}
        }

      }

      reconstTensor = tensorRDD.map{ x => (x._1, Tensor.reshapeBlock(x._1, unfoldDim, tensorInfo, new DenseMatrix[Double](x._2.numRows, x._2.numCols, x._2.values)))}
      // find cluster with best approximation for each slice
      var cluIds: Array[Int] = Array.fill(tensorInfo.tensorRank(unfoldDim))(0)
      val TapproxErr = approxErr.t
      for(i <- 0 to tensorInfo.tensorRank(unfoldDim) - 1){
        cluIds(i) = TapproxErr(::, i).argmax
      }

      // perform reconstruction inside each cluster
      for(c <- 0 to cluNum - 1){
        val clIds = cluIds.filter{ x => x == c}
        val( reconstRDD, reconstInfo ) = TensorTucker.reConst2 ( reconstTensor, clIds, coreTensors(c), coreInfo(c), deCompSeq, basisMatrices(c) )
        if(c == 0){
          reconstTensor = reconstRDD
        }/* else{
         // reconstTensor = reconstTensor + reconstRDD
        }
      }*/
      /*
      // Save reconst block header and reconst block
      Tensor.saveTensorHeader( reconstPath, reconstInfo )
      Tensor.saveTensorBlock( reconstPath, reconstRDD )
              }


              }

   {
}
*/