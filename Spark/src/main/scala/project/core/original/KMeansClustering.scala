package project.core.original

import java.lang.instrument.Instrumentation

import breeze.linalg._
import org.apache.commons.math3.ml.distance
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.util.random.XORShiftRandom

import scala.collection.mutable.ArrayBuffer
//import org.apache.spark.mllib.clustering.DistanceMeasure
import DistanceMeasure._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

class KMeansClustering (
  private var k: Int,
  private var maxIterations: Int,
  private var initializationSteps: Int,
  //private var epsilon: Double,
  private var tensorDim: Int
  /*private var seed: Long,
  private var distanceMeasure: String*/) extends Serializable with Logging {


  def train(
             data: RDD[Vector],
             k: Int,
             maxIterations: Int,
             initializationMode: String,
             seed: Long): KMeansModel = {
    new KMeansClustering().setK(k)
      .setSeed(seed)
      .run(data)
  }

  // Initial cluster centers can be provided as a KMeansModel object rather than using the
  // random or k-means|| initializationMode
  private var initialModel: Option[KMeansModel] = None

  /*
  * Compute the distance from every point of the subtensor to the centroid of each cluster
  *
   */
  def calcDistances(): Unit ={
    var distvect = Vector.zeros(this.k)

    for(i <-0 until this.k){

    }
  }

  /**
    * Train a K-means model on the given set of points; `data` should be cached for high
    * performance, because this is an iterative algorithm.
    */
  def run(data: RDD[Vector]): KMeansModel = {
    run(data, None)
  }

  private[spark] def run(
                          data: RDD[Vector],
                          instr: Option[Instrumentation]): KMeansModel = {

    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    // Compute squared norms and cache them.
    val norms = data.map(Vectors.norm(_, 2.0))
    norms.persist()
    val zippedData = data.zip(norms).map { case (v, norm) =>
      new VectorWithNorm(v, norm)
    }

    val model = runAlgorithm(zippedData, instr)
    norms.unpersist()

    // Warn at the end of the run as well, for increased visibility.
    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data was not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }
    model
  }


  /**
    * Implementation of K-Means algorithm.
    */
  private def runAlgorithm(
                            data: RDD[VectorWithNorm],
                            instr: Option[Instrumentation]): KMeansModel = {

    val sc = data.sparkContext

    val initStartTime = System.nanoTime()

    val distanceMeasureInstance = new EuclideanDistanceMeasure

    val centers = initialModel match {
      case Some(kMeansCenters) =>
        kMeansCenters.clusterCenters.map(new VectorWithNorm(_))
        initKMeansParallel(data, distanceMeasureInstance)
    }
    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9

    var converged = false
    var cost = 0.0
    var iteration = 0

    val iterationStartTime = System.nanoTime()

    instr.foreach(_.logNumFeatures(centers.head.vector.size))

    // Execute iterations of Lloyd's algorithm until converged
    while (iteration < maxIterations && !converged) {
      val costAccum = sc.doubleAccumulator
      val bcCenters = sc.broadcast(centers)

      // Find the new centers
      val collected = data.mapPartitions { points =>
        val thisCenters = bcCenters.value
        val dims = thisCenters.head.vector.size

        val sums = Array.fill(thisCenters.length)(Vectors.zeros(dims))
        val counts = Array.fill(thisCenters.length)(0L)

        points.foreach { point =>
          val (bestCenter, cost) = distanceMeasureInstance.findClosest(thisCenters, point)
          costAccum.add(cost)
          distanceMeasureInstance.updateClusterSum(point, sums(bestCenter))
          counts(bestCenter) += 1
        }

        counts.indices.filter(counts(_) > 0).map(j => (j, (sums(j), counts(j)))).iterator
      }.reduceByKey { case ((sum1, count1), (sum2, count2)) =>
        axpy(1.0, sum2, sum1)
        (sum1, count1 + count2)
      }.collectAsMap()

      if (iteration == 0) {
        instr.foreach(_.logNumExamples(collected.values.map(_._2).sum))
      }

      val newCenters = collected.mapValues { case (sum, count) =>
        distanceMeasureInstance.centroid(sum, count)
      }

      bcCenters.destroy()

      // Update the cluster centers and costs
      converged = true
      newCenters.foreach { case (j, newCenter) =>
        if (converged &&
          !distanceMeasureInstance.isCenterConverged(centers(j), newCenter, epsilon)) {
          converged = false
        }
        centers(j) = newCenter
      }

      cost = costAccum.value
      iteration += 1
    }

    val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
    logInfo(f"Iterations took $iterationTimeInSeconds%.3f seconds.")

    if (iteration == maxIterations) {
      logInfo(s"KMeans reached the max number of iterations: $maxIterations.")
    } else {
      logInfo(s"KMeans converged in $iteration iterations.")
    }
    logInfo(s"The cost is $cost.")

    new KMeansModel(centers.map(_.vector), distanceMeasure, cost, iteration)
  }

  /**
    * Initialize a set of cluster centers using the k-means|| algorithm by Bahmani et al.
    * (Bahmani et al., Scalable K-Means++, VLDB 2012). This is a variant of k-means++ that tries
    * to find dissimilar cluster centers by starting with a random center and then doing
    * passes where more centers are chosen with probability proportional to their squared distance
    * to the current cluster set. It results in a provable approximation to an optimal clustering.
    *
    * The original paper can be found at http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf.
    */
  private[clustering] def initKMeansParallel(data: RDD[VectorWithNorm],
                                             distanceMeasureInstance: DistanceMeasure): Array[VectorWithNorm] = {
    // Initialize empty centers and point costs.
    var costs = data.map(_ => Double.PositiveInfinity)

    // Initialize the first center to a random point.
    val seed = new XORShiftRandom(this.seed).nextInt()
    val sample = data.takeSample(false, 1, seed)
    // Could be empty if data is empty; fail with a better message early:
    require(sample.nonEmpty, s"No samples available from $data")

    val centers = ArrayBuffer[VectorWithNorm]()
    var newCenters = Seq(sample.head.toDense)
    centers ++= newCenters

    // On each step, sample 2 * k points on average with probability proportional
    // to their squared distance from the centers. Note that only distances between points
    // and new centers are computed in each iteration.
    var step = 0
    val bcNewCentersList = ArrayBuffer[Broadcast[_]]()
    while (step < initializationSteps) {
      val bcNewCenters = data.context.broadcast(newCenters)
      bcNewCentersList += bcNewCenters
      val preCosts = costs
      costs = data.zip(preCosts).map { case (point, cost) =>
        math.min(distanceMeasureInstance.pointCost(bcNewCenters.value, point), cost)
      }.persist(StorageLevel.MEMORY_AND_DISK)
      val sumCosts = costs.sum()

      bcNewCenters.unpersist()
      preCosts.unpersist()

      val chosen = data.zip(costs).mapPartitionsWithIndex { (index, pointCosts) =>
        val rand = new XORShiftRandom(seed ^ (step << 16) ^ index)
        pointCosts.filter { case (_, c) => rand.nextDouble() < 2.0 * c * k / sumCosts }.map(_._1)
      }.collect()
      newCenters = chosen.map(_.toDense)
      centers ++= newCenters
      step += 1
    }

    costs.unpersist()
    bcNewCentersList.foreach(_.destroy())

    val distinctCenters = centers.map(_.vector).distinct.map(new VectorWithNorm(_))

    if (distinctCenters.size <= k) {
      distinctCenters.toArray
    } else {
      // Finally, we might have a set of more than k distinct candidate centers; weight each
      // candidate by the number of points in the dataset mapping to it and run a local k-means++
      // on the weighted centers to pick k of them
      val bcCenters = data.context.broadcast(distinctCenters)
      val countMap = data
        .map(distanceMeasureInstance.findClosest(bcCenters.value, _)._1)
        .countByValue()

      bcCenters.destroy()

      val myWeights = distinctCenters.indices.map(countMap.getOrElse(_, 0L).toDouble).toArray
      LocalKMeans.kMeansPlusPlus(0, distinctCenters.toArray, myWeights, k, 30)
    }
  }
}


/**
    * Set the number of dimensions of the original tensor (tensorDim)
    *
    */
  def setTensorDim(t: Int): this.type = {
    require(t > 0, s"Number of dimensions of the original tensor must be positive but got ${t}")
    this.tensorDim = t
    this
  }


  /**
    * Set the number of clusters to create (k).
    *
    * @note It is possible for fewer than k clusters to
    * be returned, for example, if there are fewer than k distinct points to cluster. Default: 2.
    */
  def setK(k: Int): this.type = {
    require(k > 0,
      s"Number of clusters must be positive but got ${k}")
    this.k = k
    this
  }


  /**
    * Maximum number of iterations allowed.
    */
  def getMaxIterations: Int = maxIterations

  /**
    * Set maximum number of iterations allowed. Default: 20.
    */
  def setMaxIterations(maxIterations: Int): this.type = {
    require(maxIterations >= 0,
      s"Maximum of iterations must be nonnegative but got ${maxIterations}")
    this.maxIterations = maxIterations
    this
  }
}


/**
  * A vector with its norm for fast distance computation.
  */
private[clustering] class VectorWithNorm(val vector: Vector, val norm: Double)
  extends Serializable {

  def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))

  def this(array: Array[Double]) = this(Vectors.dense(array))

  /** Converts the vector to a dense vector. */
  def toDense: VectorWithNorm = new VectorWithNorm(Vectors.dense(vector.toArray), norm)
}


