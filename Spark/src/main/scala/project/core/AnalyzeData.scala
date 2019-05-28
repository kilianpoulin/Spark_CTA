package com.tp.spark.core
import com.tp.spark.utils.FlyingBus
import com.tp.spark.utils.FlyingBus.Bus
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.Logger
import org.apache.log4j.Level


case class ResultType (country : String, total : Int, reduced : Int)

object AnalyzeData extends App {

  val pathToFile = "../nuclear-flying-buses.json"

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  val conf = new SparkConf()
    .setAppName("Flying Bus data mining")
    .setMaster("local[*]")
  val sc = SparkContext.getOrCreate(conf)

  val sqlContext = new org.apache.spark.sql.SQLContext(sc)


  /**
    *  Load the data from the json file and return an RDD of Bus
    */
  def loadData(): RDD[Bus] = {
    // create spark configuration and spark context


    sc.textFile(pathToFile)
      .mapPartitions(FlyingBus.parseFromJson(_))

  }

  /* Print all data */

  // loadData().collect().foreach(println)

  /* ***********************************************************************
  ************************************************************************
  **************************** ANALYSIS ********************************
  ********************************************************************
  ******************************************************************
   */


  // get number of buses / country
  val alldata : RDD[(String, Int)] = loadData().groupBy(_.country.name).mapValues(_.size)
  val df = sqlContext.createDataFrame(alldata).toDF("country", "total")


  // Are there more broken buses when the weather is hot or cold ?
  println(" -------- Are there more broken buses when the weather is hot or cold ? -------- ")
  loadData().filter(_.broken == true).groupBy(_.weather).mapValues(_.size).foreach(println)

  println("\n")

  // Are there more broken buses in the northern or the southern hemisphere ?
  println(" -------- Are there more broken buses in the northern hemisphere ? -------- ")
  loadData().filter(_.broken == true).groupBy(_.country.northHemisphere).mapValues(_.size).sortBy(_._1, false).foreach(println)

  println("\n")

  // Among the broken buses, how many are because of an empty fuel tank
  println(" -------- Amoung the broken buses, how many are because of an empty fuel tank ? -------- ")
  println(" => Number : " + loadData().filter(tuple => (tuple.broken == true && tuple.fuel == 0)).count())

  println("\n")

  // Which lines have more full buses ?
  println(" -------- Which lines have more full buses ? -------- ")
  loadData().filter(_.passengers == 50).groupBy(_.line).mapValues(_.size).sortBy(_._1, false).foreach(println)

  println("\n")

  // Which countries have the highest ratio of full buses ?
  println(" -------- Which countries have the highest ratio of full buses  ? -------- ")

  // get number of full buses / country
  val reduceddata : RDD[(String, Int)] = loadData().filter(_.passengers == 50).groupBy(_.country.name).mapValues(_.size)
  val redf = sqlContext.createDataFrame(reduceddata).toDF("country", "reduced")

  // join data frames to compare them
  val joinedDF = df.join(redf, Seq("country"))
  val resultDF = joinedDF.join(joinedDF, joinedDF("country") === joinedDF("country")).select(joinedDF("country"), (joinedDF("reduced") / joinedDF("total")) * 100)
  resultDF.collect.foreach(println)

  println("\n")

  // Which countries have the highest ratio of broken buses ?
  println(" -------- Which countries have the highest ratio of broken buses  ? -------- ")
  // get number of full buses / country
  val reduceddata2 : RDD[(String, Int)] = loadData().filter(_.broken == true).groupBy(_.country.name).mapValues(_.size)
  val redf2 = sqlContext.createDataFrame(reduceddata2).toDF("country", "reduced")

  // join data frames to compare them
  val joinedDF2 = df.join(redf2, Seq("country"))
  val resultDF2 = joinedDF2.join(joinedDF2, joinedDF2("country") === joinedDF2("country")).select(joinedDF2("country"), (joinedDF2("reduced") / joinedDF2("total")) * 100)
  resultDF2.collect.foreach(println)

  println("\n")


  // What is the average total kms of broken buses ?
  println(" -------- What is the average total kms of broken buses ? -------- ")
  println(" => Average kms : " + loadData().filter(_.broken == true).map(_.totalKms).mean() + " kilometers")

  println("\n")

  // What is the average total kms of broken buses in the northern hemisphere ?
  println(" -------- What is the average total kms of broken buses in the northern hemisphere ? -------- ")
  println(" => Average kms : " + loadData().filter(tuple => (tuple.broken == true && tuple.country.northHemisphere == true)).map(_.totalKms).mean() + " kilometers")

  print("\n")

  // What is the average total kms of broken buses in the southern hemisphere ?
  println(" -------- What is the average total kms of broken buses in the southern hemisphere ? -------- ")
  println(" => Average kms : " + loadData().filter(tuple => (tuple.broken == true && tuple.country.northHemisphere == false)).map(_.totalKms).mean() + " kilometers")


  println("\n\n")
  /* ***********************************************************************
  ************************************************************************
  **************************** SIMPLE QUERIES **************************
  ********************************************************************
  ******************************************************************
   */

  // Number of nuclear flying buses in the northern hemisphere
  println("Number of nuclear flying buses in the northern hemisphere : " + loadData().filter(_.country.northHemisphere == false).count())

  println("\n")

  // Number of nuclear flying buses in the southern hemisphere
  println("Number of nuclear flying buses in the southern hemisphere : " + loadData().filter(_.country.northHemisphere == true).count())

  println("\n")

  println(" => List of nuclear flying buses in the northern hemisphere : ")
  // loadData().groupBy(_.country.northHemisphere == false).foreach(println)

  println("\n")

  // Nuclear Flying buses sorted by kms
  println(" -------- Nuclear flying buses sorted by kms --------- ")
  // loadData().sortBy(_.totalKms, ascending = false).foreach(println)

}