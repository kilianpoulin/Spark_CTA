package project.core.TensorPack

import project.core.Tucker.Tensor
import breeze.linalg.DenseMatrix
import org.apache.spark.rdd.RDD
import scala.collection.{mutable => CM}
import breeze.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.{DenseMatrix => DMatrix}

object GetTensor {

  def extractTensor(tensorPath: String): (RDD[(CM.ArraySeq[Int], DMatrix)], Tensor.TensorInfo) ={

    val tensorInfo = Tensor.readTensorHeader(tensorPath)

    // Read tensor block and transform into RDD
    val tensorRDD = Tensor.readTensorBlock(tensorPath, tensorInfo.blockRank)

    val dmat = tensorRDD.map { case (x, y) => y }.take(1)

    (tensorRDD, tensorInfo)
  }

  def unfoldTensor(tensorRDD: RDD[(CM.ArraySeq[Int], DMatrix)], tensorInfo: Tensor.TensorInfo, unfoldDim: Int): RDD[(CM.ArraySeq[Int], DenseMatrix[Double])] ={

    // Unfold the tensor -- pre-process before applying K-Means
    Tensor.TensorUnfoldBlocks2(tensorRDD, unfoldDim, tensorInfo.tensorRank, tensorInfo.blockRank, tensorInfo.blockNum)
  }

}
