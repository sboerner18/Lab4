name := "SparkDataframe"
scalaVersion := "2.11.8"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.3.2",
  "org.apache.spark" %% "spark-mllib" % "2.3.2" % "compile",
  "org.apache.spark" %% "spark-ml" % "2.3.2",
  "org.apache.spark" %% "spark-sql" % "2.3.2",
  "org.apache.spark" %% "spark-graphx" % "2.3.2",
  "graphframes" % "graphframes" % "0.6.0-spark2.3-s_2.11"

)
