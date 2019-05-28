package com.tp.spark.utils
import com.google.gson._
import Country.Country

object FlyingBus {

  case class Bus (
                 busId : Int,
                 fuel : Int,
                 seats : Int,
                 passengers : Int,
                 line : Int,
                 nextStop : Int,
                 nextStopDistance : Int,
                 totalKms : Int,
                 broken : Boolean,
                 weather : String,
                 country : Country
                 )


  def parseFromJson(lines:Iterator[String]):Iterator[Bus] = {
    val gson = new Gson
    lines.map(line => gson.fromJson(line, classOf[Bus]))
  }
}