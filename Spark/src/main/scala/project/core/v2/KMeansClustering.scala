package project.core.v2

import java.lang.Math.{pow, sqrt}

import org.apache.spark.rdd.RDD

import scala.math.Ordering
import scala.collection.{mutable => CM}
import org.apache.spark.mllib.linalg.{DenseMatrix => DMatrix}

import scala.util.Random
import scala.{Vector => Vect}
import project.core.v2.RunTucker.{tensorInfo, tensorRDD}
import project.core.v2.Tensor.{MyPartitioner, localTensorUnfoldBlock}
import breeze.linalg.{sum, DenseMatrix => BMatrix}
import org.apache.spark.storage.StorageLevel.{DISK_ONLY, MEMORY_AND_DISK, MEMORY_ONLY}

import scala.collection.mutable.ListBuffer
class KMeansClustering (
                         private var k: Int,
                         private var maxIterations: Int,
                         private var tensorDim: Int) extends Serializable {

  /** -----------------------------------------------------------------------------------------------------------------
    * Implicit class to add function distanceTo to Vector class
    * -----------------------------------------------------------------------------------------------------------------
    * */
  @transient case class VData(@transient v1: Vect[Double]) extends Serializable {
    @transient val distanceTo = (v2: Vect[Double]) => {
      val s: Array[Double] = v1.toArray
      //val s = Array.fill(34116)(0.0)
      val t: Array[Double] = v2.toArray
      var tmp = (s zip t).map{ case (x,y) => x - y }
      var res = tmp.map{ case(x) => pow(x, 2)}
      var resum = res.sum
      //var resroot = sqrt(resum)

      //var resfull = sqrt((s zip t).map{ case (x,y) => pow(x - y, 2) }.sum)
      resum
      //sqrt((s zip t).map{ case (x,y) => pow(x - y, 2) }.sum)
    }
  }

  @transient implicit def distancecalc(@transient v: Vect[Double]): VData = VData(v)

  def sum(list: List[Double]): Double = list match {
    case Nil => 0
    case x :: xs => x + sum(xs)
  }

  def train(
             data: RDD[ (CM.ArraySeq[Int], BMatrix[Double]) ]
           ) = {
    run(data)
  }

  /**
    * Set the number of clusters to create (k).
    *
    */
  def setK(k: Int): this.type = {
    require(k > 0,
      s"Number of clusters must be positive but got ${k}")
    this.k = k
    this
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Main function for K-Means
    * Contains all steps
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def getTupleList(list: List[(List[Double], Int)])={
    var newlist = list
    newlist.toArray.map{ case(l, id) => list.filter{ case(l1, id1) => id1 == id}.map(_._1)}.distinct.map{ case(x) => x.transpose.map(_.sum)}.flatMap(x => x).toList
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Combines and addition partial distances to get the distance to each centroid for a full vector
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def combinePartialDistances(list: List[Array[(Vect[Double], Array[Double])]]) ={
    list.toArray.map{ case(x) => x.map{ case (y, z) => z.toList}.zipWithIndex.toList}.toList.flatMap(s => s).groupBy(e => e._2).map(_._2).map{ case(x) => getTupleList(x)}
    //map{case(a) => a.toArray.map(_._1).flatten}.transpose.map(_.sum)
    //.map(_._1)
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Vectors are cut into partial vectors, each in different blocks
    * This function finds all blocks by their ID, that contain a partial vector of a vector.
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def getVectorIds(ids: CM.ArraySeq[Int]): List[CM.ArraySeq[Int]] ={
    var listIds = new Array[CM.ArraySeq[Int]](tensorInfo.blockNum(1))
    for(i <- 0 until tensorInfo.blockNum(1)){
      listIds(i) = ids.clone()
      listIds(i)(1) = i
    }
    listIds.toList
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Returns a vector in a matrix, based on its row number
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def getMatVect(data: BMatrix[Double], row: Int): Vect[Double] ={
    var tmpArray = new Array[Double](data.cols)
    for(i <- 0 until data.cols){
      tmpArray(i) = data.apply(row, i)
    }
    tmpArray.toVector
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Initializes partial centroids for each cluster
    * Each partial centroid is in fact a randomly chosen vector in the corresponding block
    * nbIt is the number of iterations we want to do in order to try to find a vector that different from the vector 0
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def createRandomCentroids(data: BMatrix[Double], option: String = "") = {
    var centroids: Array[Vect[Double]] = new Array[Vect[Double]](k)
    if(option == "default"){
      var cluster1 = MySpark.sc.textFile("../Spark/data/clustertmp/cluster1spark")
      var num = cluster1.getNumPartitions

      val parts = cluster1.mapPartitionsWithIndex((idx, iter) => if (idx == 0) iter else Iterator()).collect()

      //cluster1.persist(MEMORY_ONLY)
      var tmp11 =  parts.map{ case(x) => x.split(",")}
      var tmp12 = parts.map{ case(x) => x.toDouble}
      /*var tmp12 = cluster1.foreachPartition(partition => {
        if(partition != null)
          partition.map{ case(x) => x.split(",")}.toList.foreach(println)
      })*/

      var cluster2 = MySpark.sc.textFile("../Spark/data/clustertmp/cluster2spark")
      cluster2.persist(MEMORY_AND_DISK)
      var tmp21 = cluster2.collect().map{ case(x) => x.split(",")}
      var tmp22 = tmp21(0).map{ case(x) => x.toDouble}

      val cluster3 = MySpark.sc.textFile("../Spark/data/clustertmp/cluster3spark")
      cluster3.persist(MEMORY_AND_DISK)
      var tmp31 = cluster3.collect().map{ case(x) => x.split(",")}
      var tmp32 = tmp31(0).map{ case(x) => x.toDouble}

     // centroids(0) = cluster1.map{ case(x) => x.toDouble}.toVector
      //centroids(0).foreach(println)
      //centroids(0) = tmp12.toVector
      centroids(1) = tmp22.toVector
      centroids(2) = tmp32.toVector
      //centroids(1) = cluster2.map{ case(x) => x.toDouble}.toVector
      //centroids(2) = cluster3.map{ case(x) => x.toDouble}.toVector
      //centroids.foreach(println)
    } else {
      val randomIndices = ListBuffer[Int]()
      val random = new Random()
      var tmp = 0
      var nbIt = 0
      while (randomIndices.size < k) {
        // choosing one of the vectors in the dataset to become one cluster centroid
        // so we will choose k vectors in the dataset
        // we try to only choose vectors not equal to zero vector
        tmp = random.nextInt(data.rows.toInt)
        //println("sum = " + data.take(tmp)(0).toArray.sum)
        if (getMatVect(data, tmp).toArray.sum != 0 || nbIt > 5) {
          randomIndices += tmp
          nbIt = 0
        } else {
          nbIt += 1
        }
      }

      var centroids: Array[Vect[Double]] = new Array[Vect[Double]](randomIndices.size)
      for (i <- 0 until randomIndices.size)
        centroids(i) = getMatVect(data, randomIndices(i))
    }
    centroids
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Returns the distance from one vector to each centroid vector
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def calcDistances(centroids: Array[Vect[Double]], vector: Vect[Double]): Array[Double] ={
    centroids.map{ case(c) => vector.distanceTo(c)}
  }
  def calcDistances2(centroids: Vect[Double], vector: Vect[Double]): Double ={
    vector.distanceTo(centroids)
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Returns a tuple containing a vector and all the distances from this vector to each centroid vector
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def getPartialDistances(ids: CM.ArraySeq[Int], centroids: Array[Vect[Double]], data:Array[Vect[Double]]) = {
    data.map{
      case(vect) => calcDistances(centroids, vect)
    }
  }

  def getPartialDistances2(ids: CM.ArraySeq[Int], centroids: Vect[Double], data:Array[Vect[Double]]) = {
    calcDistances2(centroids, data(0))
  }


  /** -----------------------------------------------------------------------------------------------------------------
    * Main function for K-Means
    * Contains all steps
    * -----------------------------------------------------------------------------------------------------------------
    * */
  def run(data: RDD[ (CM.ArraySeq[Int], BMatrix[Double]) ]){

    // Step 1 : Unfold each block and make one partition for one block
    /*val newdata = data.map{ case(ids, mat) =>
      (ids, localTensorUnfoldBlock( mat, ids, 0,  tensorInfo.blockRank, tensorInfo.blockNum, tensorInfo.tensorRank))
    }.reduceByKey( new MyPartitioner( tensorInfo.blockNum ), ( a, b ) => a + b )
*/
    var newdata = data
    println(" (3) [OK] Tensor unfolded along Dimension 1 ")

    // Step 2 : create k clusters with random values as centroids data)

    /*val cluster1 = MySpark.sc.textFile("../Spark/data/clustertmp/cluster1spark")
    cluster1.foreach(println)*/
   /* var centroids = newdata.map{ case(ids, mat) => (ids, createRandomCentroids(mat, "default"))}
centroids.foreach(println)
    println(" (4) [OK] " + k + " splitted clusters successfully initialized ")
*/
    var cent1 = MySpark.sc.textFile("../Spark/data/clustertmp/cluster1full").collect()
    var cent2 = MySpark.sc.textFile("../Spark/data/clustertmp/cluster2full").collect()
    var cent3 = MySpark.sc.textFile("../Spark/data/clustertmp/cluster3full").collect()
    var centRDD = MySpark.sc.parallelize(Array(cent1, cent2, cent3))
    /*var count1 = centroids.map{ case(x) => x.split(",")}.flatMap(x => x).count()
    println("count = " + count1)*/
    //var centroids = centRDD.map{ case(cent) => cent.map{ case(x) => x.split(",")}.flatMap(x => x).map{ case(y) => y.toDouble}.toList.grouped(9800).map{case (z) => z.toVector}.toArray}.collect()
    var centroids = centRDD.map{ case(cent) => cent.map{ case(x) => x.split(",")}.flatMap(x => x).map{ case(y) => y.toDouble}.toList.grouped(34116).map{case (z) => z.toVector}.toArray}.collect()

    //centroids.take(1).map{case (x) => x.toList}.toList.foreach(println)

  //  var centdata = data.map{ case(ids, map) => ids}.foreach(println)
    /*test.take(1).foreach(println)
    test.take(2).foreach(println)

    test.take(3).foreach(println)

    test.take(4).foreach(println)

    test.take(5).foreach(println)
    test.take(7).foreach(println)*/
    println("success")

    //newdata = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](0,0)}
    // only 4
    var mat00 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](0,0)}.map{ case(ids, mat) => mat}.collect()
    var mat01 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](0,1)}.map{ case(ids, mat) => mat}.collect()
    var mat02 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](0,2)}.map{ case(ids, mat) => mat}.collect()
    var mat03 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](0,3)}.map{ case(ids, mat) => mat}.collect()
    var mat031 = mat03(0)(::,(0 to 4715).toIndexedSeq)
    var block11 = BMatrix.horzcat(mat00(0), mat01(0), mat02(0), mat031)


    var mat04 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](0,3)}.map{ case(ids, mat) => mat}.collect()
    var mat05 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](0,4)}.map{ case(ids, mat) => mat}.collect()
    var mat06 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](0,5)}.map{ case(ids, mat) => mat}.collect()
    var mat07 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](0,6)}.map{ case(ids, mat) => mat}.collect()
    var mat041 = mat04(0)(::,(4716 to 9799).toIndexedSeq)
    var block12 = BMatrix.horzcat(mat041, mat05(0), mat06(0), mat07(0))

    var mat10 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](1,0)}.map{ case(ids, mat) => mat}.collect()
    var mat11 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](1,1)}.map{ case(ids, mat) => mat}.collect()
    var mat12 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](1,2)}.map{ case(ids, mat) => mat}.collect()
    var mat13 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](1,3)}.map{ case(ids, mat) => mat}.collect()
    var mat131 = mat13(0)(::,(0 to 4715).toIndexedSeq)
    var block21 = BMatrix.horzcat(mat10(0), mat11(0), mat12(0), mat131)


    var mat14 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](1,3)}.map{ case(ids, mat) => mat}.collect()
    var mat15 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](1,4)}.map{ case(ids, mat) => mat}.collect()
    var mat16 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](1,5)}.map{ case(ids, mat) => mat}.collect()
    var mat17 = newdata.filter{ case(ids, mat) => ids == CM.ArraySeq[Int](1,6)}.map{ case(ids, mat) => mat}.collect()
    var mat141 = mat14(0)(::,(4716 to 9799).toIndexedSeq)
    var block22 = BMatrix.horzcat(mat141, mat15(0), mat16(0), mat17(0))

    var fulldata = MySpark.sc.parallelize(Array((CM.ArraySeq[Int](0,0), block11),
      (CM.ArraySeq[Int](0,1), block12),
      (CM.ArraySeq[Int](1,0), block21),
      (CM.ArraySeq[Int](1,1), block22)))

    println("end test")
    var test3 = fulldata.map{ case(ids, values) => (ids, getPartialDistances(ids, centroids.flatMap{case (x) => x}, Tensor.blockTensorAsVectors(values)))}.collect()
    var distances = test3.map{ case(ids, dist) => dist.toList}
    println("done")
    //test3.take(1).foreach(println)
    return
    /*
    // Step 3 : calculate distance between partial vectors and each partial centroid vector
  //  var distances = centroids.join(newdata).map { case (ids, (centroids, values)) => (ids, getPartialDistances(ids, centroids, Tensor.blockTensorAsVectors(values))) }
    println(" (5) [OK] calculating distances between partial vectors and each partial centroid vector ")
    // display list of distances
    //distances.map{ case(id, values) => values }.flatMap(v => v).map{ case(vector, distances) => distances}.flatMap(a => a).collect().foreach(println)


    // Step 4 : Addition partial distances
    // adding partial distances
    var tmpDistances = distances.map{ x => (getVectorIds(x._1), x._2) }.groupByKey().map{case (i, iter) => (i, iter.toList)}
    var fullDistances = tmpDistances.map{ case(i, iter) => (i, combinePartialDistances(iter))}

    //fullDistances.foreach(println)
    fullDistances.saveAsTextFile("testdist")

    //test.map{ case(i, iter) => combinePartialVectors(iter).size}.foreach(println)
    println(" (6) [OK] Addition partial distances for each full vector ")

    /* IN PROGRESS
    // Step 5 : Build clusters based on the shortest distance of the full vector
      // we need to break the ids of fullDistances
      var tmpfullDistances = fullDistances.map{case(x) => (x._1, x._2)}.flatMap{ case(a,b) => a.map(y => (y, b))}
      //println("test = " + test.size)
      var tmpclusters = centroids.join(tmpfullDistances).join(distances.map{ case(ids, x) => (ids, x.map{ case(centroids, vectors) => (centroids, vectors)})}).collect
        var clusters = tmpclusters
        .map{ case(ids, o) => (ids, buildClusters(ids, o._1._1, o._2.map{ case(c, d) => c},  o._1._2))}
    // centroids, vectors, distances
      //var clusters = centroids.join(newdata).map{ case(ids, (centroids, values)) => (ids, buildClusters(ids, centroids, values))}

      println(" (7) [OK] Building clusters ")
      */
*/
  }

  /** -----------------------------------------------------------------------------------------------------------------
    * Dispatches partial vectors into each partial cluster
    * -----------------------------------------------------------------------------------------------------------------
    * */
  /*
  def buildClusters(ids: CM.ArraySeq[Int], centroids: Array[Vect[Double]], data: Array[Vect[Double]], distances: Iterable[List[Double]]) = {
    println(ids)
    //val data = newel.filter{ case(id, _) => id == ids}.map{ case(id, mat) => mat}.take(1)

    //var newdata = Tensor.blockTensorAsVectors(data)
    //println("count data = " + newdata.count())
    data.map({ vect =>
      val byDistanceToCentroid = new Ordering[Double] {
        def compare(v1: Double, v2: Double) = v1 compareTo v2
      }
      (vect, centroids.map({ case(centroid) => centroid }) min byDistanceToCentroid)
    })
      .groupBy({ case(centroid, _) => centroid }).map({ case(centroid, vectorsToCentroids) =>
        val vects = vectorsToCentroids.map({ case(vect, _) => vect })
        (centroid, vects)
      })
  }
  /*
  // returns Map[Vector, RDD[Vector]]
  //@tailrec
  def buildClusters(data: RDD[Vector], prevClusters: scala.collection.Map[Int, (Vector, RDD[Vector])]): scala.collection.Map[Vector, Iterable[Vector]] = {

    println("------------------- Build clusters -------------------- ")
    // get rid of I
    /*val nextClusters = data.map({ vect =>
      val byDistanceToPoint = new Ordering[Vector] {
        //override def compare(v1: Vector, v2: Vector) = distanceTo(v1, vect) compareTo distanceTo(v2, vect)
        @transient def compare(v1: Vector, v2: Vector) = v1.distanceTo(vect) compareTo v2.distanceTo(vect)
      }
      (vect, prevClusters.map({ case (_,(centroid, _)) => centroid }) min byDistanceToPoint)
    }).*/groupBy({ case (centroid, _) => centroid }).map({ case (centroid, pointsToCentroids) =>
      val vects = pointsToCentroids.map({ case (vect, _) => vect })
      (centroid, vects)
    })

    println("---------- end clustering ------ ")

/*
    if (prevClusters != nextClusters) {
      val nextClustersWithBetterCentroids = nextClusters.map({
        case (centroid, members) =>
          val (sum, count) = members.foldLeft((Vector(0, 0, 0), (0))({ case ((acc, c), curr) => (acc sum curr.t, c + 1) })
          (sum divideBy count, members)
      })

      buildClusters(data, nextClustersWithBetterCentroids)
    } else {*/
      //prevClusters
    nextClusters.collectAsMap()
   // }
  }
*/
* */
}