package com.tp.spark.core



import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

/**
  *  Here the goal is to count how much each word appears in a file and make some operation on the result.
  *  We use the mapreduce pattern to do this:
  *
  *  step 1, the mapper:
  *  - we attribute 1 to each word. And we obtain then couples (word, 1), where word is the key.
  *
  *  step 2, the reducer:
  *  - for each key (=word), the values are added and we will obtain the total amount.
  */
object Test extends App {

  val pathToFile = "data/wordcount.txt"

  /**
    *  Load the data from the text file and return an RDD of words
    */
  def loadData(): RDD[String] = {
    // create spark configuration and spark context: the Spark context is the entry point in Spark.
    // It represents the connexion to Spark and it is the place where you can configure the common properties
    // like the app name, the master url, memories allocation...
    val conf = new SparkConf()
      .setAppName("Wordcount")
      .setMaster("local[*]") // here local mode. And * means you will use as much as you have cores.

    val sc = SparkContext.getOrCreate(conf)

    // load data and create an RDD where each element will be a word
    // Here the flatMap method is used to separate the word in each line using the space separator
    // In this way it returns an RDD where each "element" is a word
    sc.textFile(pathToFile)
  }

  //loadData().foreach(println)
  wordcount().foreach(println)

  /**
    *  Now count how much each word appears!
    */
  def wordcount(): RDD[(String, Int)] = {
    loadData()
      .flatMap(_.split(" ")).map(word => (word, 1)).reduceByKey(_+_)

    //.reduceByKey((acc, i) => acc = i)
  }

}
