import java.io.File
import java.lang.Math.{pow, sqrt}

import scala.annotation.tailrec
import scala.util.Random

case class Point(x: Double, y: Double, z: Double) {
  def distanceTo(that: Point) = sqrt(pow(this.x - that.x, 2) + pow(this.y - that.y, 2) + pow(this.z - that.z, 2))

  def sum(that: Point) = Point(this.x + that.x, this.y + that.y, this.z + that.z)

  def divideBy(number: Int) = Point(this.x / number, this.y / number, this.z / number)

  override def toString = s"$x,$y,$z"
}

object KMeansClustering {

  val K = 4

  def main(args: Array[String]) {
    val points = read("input.txt")
    val clusters = buildClusters(points, createRandomCentroids(points))
    clusters.foreach({
      case (centroid, members) =>
        members.foreach({ member => println(s"Centroid: $centroid Member: $member") })
    })
  }

  def read(path: String): List[Point] = {
    scala.io.Source
      .fromFile(new File(path))
      .getLines()
      .map(_.split("\\t"))
      .map({ tokens => Point(tokens(0).toDouble, tokens(1).toDouble, tokens(2).toDouble) })
      .toList
  }

  def createRandomCentroids(points: List[Point]): Map[Point, List[Point]] = {
    val randomIndices = collection.mutable.HashSet[Int]()
    val random = new Random()
    while (randomIndices.size < K) {
      randomIndices += random.nextInt(points.size)
    }

    points
      .zipWithIndex
      .filter({ case (_, index) => randomIndices.contains(index) })
      .map({ case (point, _) => (point, Nil) })
      .toMap
  }

  @tailrec
  def buildClusters(points: List[Point], prevClusters: Map[Point, List[Point]]): Map[Point, List[Point]] = {
    val nextClusters = points.map({ point =>
      val byDistanceToPoint = new Ordering[Point] {
        override def compare(p1: Point, p2: Point) = p1.distanceTo(point) compareTo p2.distanceTo(point)
      }

      (point, prevClusters.keys min byDistanceToPoint)
    }).groupBy({ case (_, centroid) => centroid })
      .map({ case (centroid, pointsToCentroids) =>
        val points = pointsToCentroids.map({ case (point, _) => point })
        (centroid, points)
      })

    if (prevClusters != nextClusters) {
      val nextClustersWithBetterCentroids = nextClusters.map({
        case (centroid, members) =>
          val (sum, count) = members.foldLeft((Point(0, 0, 0), 0))({ case ((acc, c), curr) => (acc sum curr, c + 1) })
          (sum divideBy count, members)
      })

      buildClusters(points, nextClustersWithBetterCentroids)
    } else {
      prevClusters
    }
  }

}