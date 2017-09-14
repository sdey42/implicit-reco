name := "rec-engine"

version := "1.0"

scalaVersion := "2.11.8"

//Cmd+Alt+Y to force project refresh
libraryDependencies ++= Seq(
  "org.scala-lang" % "scala-library" % "2.11.8",
  "org.scala-lang" % "scala-reflect" % "2.11.8",
  "org.apache.spark" %% "spark-core" % "2.1.0",
  "org.apache.spark" %% "spark-sql" % "2.1.0",
  "org.apache.spark" %% "spark-mllib" % "2.1.0"
)

