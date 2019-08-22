package project.core.Tucker

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

/**.
  * Creating MySpark object to set only one SparkContext
  * */
object MySpark{

  // Set Spark context
  val conf = new SparkConf()
    .setAppName("TensorTucker").setMaster("local[*]")

  val sc = new SparkContext( conf )
  TensorTucker.setSparkContext( sc )
  Tensor.setSparkContext( sc )


  val Spark = SparkSession
    .builder()
    .appName("CTA Algorithm")
    .getOrCreate()
}
