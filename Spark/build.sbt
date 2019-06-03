name := "Spark Project"

version := "1.0"


scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.0"
libraryDependencies += "com.fasterxml.jackson.core" % "jackson-databind" % "2.6.5"
libraryDependencies += "com.google.code.gson" % "gson" % "2.7"

//libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.0"

libraryDependencies ++= Seq("com.github.fommil.netlib" % "all" % "1.1.1" pomOnly())
//libraryDependencies += "org.apache.spark" %% "spark-ml" % "2.1.0"
/*
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core_2.10" % "1.1.0" % "provided",
  "org.apache.spark" %% "spark-mllib_2.10" % "1.1.0"
)*/

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2",
  "org.scalanlp" %% "breeze-viz" % "0.11.2"
)

resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

