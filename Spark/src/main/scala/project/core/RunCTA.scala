package project.core

import org.apache.log4j.{Level, Logger}
import org.apache.spark
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd._
import org.apache.spark.sql.SparkSession
import project.core.Tensor._


object RunCTA extends App {
  //header/part-00000
  // data/donkey_spark/header/part-00000.txt

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)


  val conf = new SparkConf()
    .setAppName("CTA algorithm")
    .setMaster("local[*]")

  val sc = SparkContext.getOrCreate(conf)
  val Spark = SparkSession
    .builder()
    .appName("CTA Algorithm")
    .getOrCreate()

  import Spark.implicits._


  //val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  // /!\ what is that ?
  var rddPartitionSize = 10

  val CluNum = 2 // number of clusters


  var test = Tensor.createNullTensorInfo()
  var tensorInfo = readTensorHeader("data/donkey_spark/header/part-00000")


  var tensorRDD = readTensorBlock("data/donkey_spark/block/part-00000", tensorInfo.blockRank, rddPartitionSize.toString)

  // ----------------------------------------------------
  // Clustering stage
  // ------------------------------------------------------------
  // perform K-Means
  var parsedData = tensorRDD.map(s => s._2) // get an RDD of DenseMatrices

  // testing to create a dense matrix
  // 1.0  1.0
  // 2.0  1.0
  // 4.0  3.0
  // 5.0  4.0
  val myArray = Array[Double](1.0, 2.0, 4.0, 5.0, 1.0, 1.0, 3.0, 4.0)
  var mat = new DenseMatrix(4, 2, myArray)
  //print(mat)

  // converting matrix into RDD[Vector]
  // [1.0, 2.0]
  // [3.0, 4.0]
  // [5.0, 6.0]
  var myRDDVect = matrixToRDD(mat)
  myRDDVect.collect().foreach(println)

  var numIterations = 0


  // perform KMeans
  var clusters = KMeans.train(myRDDVect, CluNum, numIterations)


  // Print clusters centers
  println("--------- CENTER OF EACH CLUSTER -------------------")
  clusters.clusterCenters.foreach(println)


    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(myRDDVect)
    println("Within Set Sum of Squared Errors = " + WSSSE)
/*
      // Save and load model
      clusters.save(sc, "KMeansModeltest5")
      val sameModel = KMeansModel.load(sc, "KMeansModeltest5")

      val foo = sameModel.predict(myRDDVect)
      foo.toDF.show()*/


      //tensorRDD.take(120).foreach(println)
  println("CTA Success !!!!!")

}
