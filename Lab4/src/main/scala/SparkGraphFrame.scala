import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.graphframes.GraphFrame
import org.apache.spark.rdd.RDD
/*import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification*/
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.{VectorUDT, Vectors}
import scala.util.MurmurHash
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils

object SparkGraphFrame {
  class SimpleCSVHeader(header:Array[String]) extends Serializable {
    val index = header.zipWithIndex.toMap
    def apply(array:Array[String], key:String):String = array(index(key))
  }

  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir", "C:\\winutils")
    val spark = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .config("spark.master", "local")
      .getOrCreate()





   //Naive Bayes Implementation

    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(spark.sparkContext, "sample_libsvm_data.txt")

    // Split data into training (60%) and test (40%).
    val Array(training, test) = data.randomSplit(Array(0.6, 0.4))

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    // Save and load model
    model.save(spark.sparkContext, "target/tmp/myNaiveBayesModel")
    val sameModel = NaiveBayesModel.load(spark.sparkContext, "target/tmp/myNaiveBayesModel")


    //Decision Tree Implementation

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val model2 = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = test.map { point =>
      val prediction = model2.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Error = $testErr")
    //println(s"Learned classification tree model:\n ${model.toDebugString}")

    // Save and load model
    model2.save(spark.sparkContext, "target/tmp/myDecisionTreeClassificationModel")
    val sameModel2 = DecisionTreeModel.load(spark.sparkContext, "target/tmp/myDecisionTreeClassificationModel")



    //Random Forest Implementation

    val numClasses2 = 2
    val categoricalFeaturesInfo2 = Map[Int, Int]()
    val numTrees2 = 3 // Use more in practice.
    val featureSubsetStrategy2 = "auto" // Let the algorithm choose.
    val impurity2 = "gini"
    val maxDepth2 = 4
    val maxBins2 = 32

    val model3 = RandomForest.trainClassifier(training, numClasses2, categoricalFeaturesInfo2,
      numTrees2, featureSubsetStrategy2, impurity2, maxDepth2, maxBins2)

    // Evaluate model on test instances and compute test error
    val labelAndPreds2 = test.map { point =>
      val prediction = model3.predict(point.features)
      (point.label, prediction)
    }
    val testErr2 = labelAndPreds2.filter(r => r._1 != r._2).count.toDouble / test.count()
    println(s"Test Error = $testErr2")
    //println(s"Learned classification forest model:\n ${model.toDebugString}")

    // Save and load model
    model3.save(spark.sparkContext, "target/tmp/myRandomForestClassificationModel")
    val sameModel3 = RandomForestModel.load(spark.sparkContext, "target/tmp/myRandomForestClassificationModel")



    import spark.implicits._
    /*val input = spark.createDataFrame(List(
      ("a", "Alice", 34),
      ("b", "Bob", 36),
      ("c", "Charlie", 30),
      ("d", "David", 29),
      ("e", "Esther", 32),
      ("f", "Fanny", 36),
      ("g", "Gabby", 60)
    )).toDF("id", "name", "age")
    val output = spark.createDataFrame(List(
      ("a", "b", "friend"),
      ("b", "c", "follow"),
      ("c", "b", "follow"),
      ("f", "c", "follow"),
      ("e", "f", "follow"),
      ("e", "d", "friend"),
      ("d", "a", "friend"),
      ("a", "e", "friend")
    )).toDF("src", "dst", "relationship")

    val g = GraphFrame(input,output)
    g.vertices.show()
    g.edges.show()*/






  /*  val data = spark.sparkContext.textFile("Absenteeism_at_work_AAA/Absenteeism_at_work_modified.txt")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(';').map(_.toDouble)))
    }
    // Split data into training (60%) and test (40%).
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")
    //val model2 = DecisionTree.train(training, lambda = 1.0)

    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    //print(accuracy)
    print(accuracy.getClass())
    print(predictionAndLabel.getClass())
*/

    // Save and load model

/*
    model.save(spark.sparkContext, "myModelPath")
    val sameModel = NaiveBayesModel.load(spark.sparkContext, "myModelPath")
*/




//Best Attempt with Actual Dataset

/*


    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("csv").option("header", "true").load("Absenteeism_at_work_AAA\\Absenteeism_at_work.csv")
    data.createOrReplaceTempView("data")

    val data2 = spark.sql("select ID, concat('Reason for absence','Month of absence','Day of the week','Seasons','Transportation expense','Distance from Residence to Work','Service time','Age','Work load Average/day ','Hit target','Disciplinary failure','Education','Son','Social drinker','Social smoker','Pet','Weight','Height','Body mass index','Absenteeism time in hours') as soonfeatures from data")
    data2.printSchema()

    val data3 = data2.withColumn("soonerfeatures",  split(data2("soonfeatures"), ",").cast("array<long>"))

    data3.printSchema()

    val convertToVector = udf((array: Seq[Long]) => {
      Vectors.dense(array.map(_.toDouble).toArray)
    })

    val data4 = data3.withColumn("features", convertToVector($"soonerfeatures"))
    data4.printSchema()

    //ata3.printSchema()
    //val data = spark.read.format("csv").load("Absenteeism_at_work_AAA/Absenteeism_at_work.csv")
    //val data2 = data.withColumn("features", )
    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = data4.randomSplit(Array(0.7, 0.3), seed = 1234L)

    // Train a NaiveBayes model.
    val model = new NaiveBayes()
      .fit(trainingData)

    // Select example rows to display.
    val predictions = model.transform(testData)
    predictions.show()

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test set accuracy = " + accuracy)
*/



   /*  val data = spark.read.format("csv").option("header", "true").load("Absenteeism_at_work_AAA\\Absenteeism_at_work.csv")

    data.createOrReplaceTempView("data")

    val label = spark.sql("select ID from data")

    val features = spark.sql("select 'Reason for absence','Month of absence','Day of the week','Seasons','Transportation expense','Distance from Residence to Work','Service time','Age','Work load Average/day ','Hit target','Disciplinary failure','Education','Son','Social drinker','Social smoker','Pet','Weight','Height','Body mass index','Absenteeism time in hours'")
    val nb =
*/
  /*  val input = stations.select("id", "dockcount", "landmark")
    val output = trips.select("src", "dst", "relationship")

    val g = GraphFrame(input, output)
  /*  g.vertices.show()
    g.edges.show()
*/
    //triangle
    val results = g.triangleCount.run()
    //results.select("id", "count").show()

    //bfs
    val results2= g.bfs.fromExpr("id = 'Market at Sansome'").toExpr("dockcount > 3").run()
    //results2.show()

    //pagerank
    val results3 = g.pageRank.resetProbability(0.15).tol(0.01).run()
    results3.vertices.select("id", "pagerank").show()
    results3.edges.select("src", "dst", "weight").show()


*/


    }
}