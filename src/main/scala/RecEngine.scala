package com.okurtv.recengine

import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql._
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.functions._

/**
  * Created by Sila on 2017/10/03
  *
  * Link to dataset: https://www.instacart.com/datasets/grocery-shopping-2017
  * Data dictionary: https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b
  *
  * ALS based implicit-feedback recommender system evaluated on Mean Percentile Ranking (MPR)
  */

object RecEngine {

  def getRawData(
                  spark: SparkSession,
                  fType: String,
                  inDir: String,
                  mapFtypeToFilenames: Map[String, String]
                ): DataFrame = {

    spark.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ",")
      .load(inDir + mapFtypeToFilenames(fType))
  }

  def getUserItemFreqDataset(
                              spark: SparkSession,
                              inDsPrior: Dataset[PriorOrdersDataset],
                              inDsOrders: Dataset[OrdersDataset]
                            ): Dataset[UserItemFreqDataset] = {
    // Need to join inDsOrders with inDsPrior on order_id
    import spark.implicits._
    val tmpDsUserItem = inDsPrior.joinWith(inDsOrders, inDsPrior.col("order_id") === inDsOrders.col("order_id"))
      .map(r => UserItemDataset(r._2.user_id, r._1.product_id))
      .as[UserItemDataset]

    tmpDsUserItem
      .map(x => ((x.userId, x.itemId), 1))
      .groupByKey(_._1)
      .count()
      .map(x => UserItemFreqDataset(x._1._1, x._1._2, x._2.toInt))
      .toDF("userId", "itemId", "freq")
      .as[UserItemFreqDataset]
  }

  def getTrainTestDatasets(
                            spark: SparkSession,
                            inDs: Dataset[UserItemFreqDataset])
  : (Dataset[UserItemFreqDataset], Dataset[UserItemFreqDataset]) = {
    import spark.implicits._
    val tmpDs = inDs.withColumn("rand_gen", rand()).toDF("userId", "itemId", "freq", "rand_gen")
    val window = Window.partitionBy("userId").orderBy(col("rand_gen").desc)
    val rankedDF = tmpDs.select(col("userId"), col("itemId"),
      col("freq"), percent_rank().over(window).as("percent_rank"))
    rankedDF.persist()
    rankedDF.count()

    val trainDs: Dataset[UserItemFreqDataset] = rankedDF.where(col("percent_rank") <= 0.7)
      .drop(col("percent_rank"))
      .toDF("userId", "itemId", "freq")
      .as[UserItemFreqDataset]
    val trainItemsToAdd = inDs.map(_.itemId).distinct()
      .except(trainDs.map(_.itemId).distinct()).toDF("itemId")
    val trainAddendumDs = trainItemsToAdd.join(rankedDF, Seq("itemId"))
      .where(col("percent_rank") > 0.7)
      .select(col("userId"), col("itemId"), col("freq"),
        rank().over(Window.partitionBy("itemId").orderBy(col("freq"))).as("rank_itm")
      ).where(col("rank_itm")===1).drop("rank_itm", "percent_rank")
      .toDF("userId", "itemId", "freq")
      .as[UserItemFreqDataset]
    val outTrainDs: Dataset[UserItemFreqDataset] = trainDs.union(trainAddendumDs)

    val testDFMinusUnseenRows = rankedDF.where(col("percent_rank") > 0.7)
      .drop(col("percent_rank"))
    val testUnseenRowsToAdd = inDs.select(col("userId")).distinct().
      crossJoin(
        inDs.select(col("itemId")).distinct().select(col("itemId"), rand().as("rand_itm"))
          .sort(col("rand_itm")).limit(1000).drop("rand_itm")
      ).join(inDs.select(col("userId"), col("itemId"), col("freq")).distinct(), Seq("userId", "itemId"), "left_outer")
      .where(isnull(col("freq")))
      .withColumn("freq", lit(0))
      .toDF("userId", "itemId", "freq")
    val testDs: Dataset[UserItemFreqDataset] = testDFMinusUnseenRows.union(testUnseenRowsToAdd)
      .toDF("userId", "itemId", "freq")
      .as[UserItemFreqDataset]
    val testItemsToAdd = inDs.map(_.itemId).distinct()
      .except(testDs.map(_.itemId).distinct())
      .toDF("itemId")
    val testAddendumDs = testItemsToAdd.select(col("itemId"))
      .crossJoin(
        rankedDF.select(col("userId")).distinct().select(col("userId"), rand().as("rand_usr"))
          .sort("rand_usr").limit(100).drop("rand_usr")
      )
      .map(r => (r.getInt(1), r.getInt(0)))
      .toDF("userId", "itemId")
      .join(testDs.select(col("userId"), col("itemId")), Seq("userId", "itemId"), "left_outer")
      .join(inDs.select(col("userId"), col("itemId")), Seq("userId", "itemId"), "left_outer")
      .withColumn("rand_itm", rand())
      .select(col("userId"), col("itemId"),
        rank().over(Window.partitionBy("itemId").orderBy(col("rand_itm").desc)).as("rank_itm")
      )
      .where(col("rank_itm") === 1)
      .drop("rand_itm", "rank_itm")
      .withColumn("freq", lit(0))
      .toDF("userId", "itemId", "freq").as[UserItemFreqDataset]
    val outTestDs = testDs.union(testAddendumDs)

    (outTrainDs, outTestDs)
  }

  def runALSToGetPredictions(
                              spark: SparkSession,
                              inDsTrain: Dataset[UserItemFreqDataset],
                              inDsTest: Dataset[TestDatasetForPrediction],
                              params: ALSImplicitParams
                            ): (ALSModel, Dataset[TestPredictionsDataset]) = {
    val als = new ALS()
      .setImplicitPrefs(true)
      .setSeed(42L)
      .setNumBlocks(4)
      .setNonnegative(params.nonNegativeFlag)
      .setColdStartStrategy(params.coldStartStrategy)
      .setMaxIter(params.maxIters)
      .setRank(params.numLatentFactors)
      .setAlpha(params.alpha)
      .setRegParam(params.lambdaForReg)
      .setFinalStorageLevel("MEMORY_AND_DISK_SER")
      .setUserCol("userId")
      .setItemCol("itemId")
      .setRatingCol("freq")
      .setPredictionCol("prediction")

    println(s"***** START - Fitting ALS Model to Train *****")
    val model: ALSModel = als.fit(inDsTrain)
    println(s"***** END - Fitting ALS Model to Train *****")

    println(s"***** START - Making Test Predictions *****")
    import spark.implicits._
    val predictions: Dataset[TestPredictionsDataset] = model.transform(inDsTest)
      .toDF("userId", "itemId", "prediction")
      .as[TestPredictionsDataset]
    println(s"***** END - Making Test Predictions *****")
    (model, predictions)
  }

  def getMPRScore(
                   spark: SparkSession,
                   inDsTest: Dataset[UserItemFreqDataset],
                   inDsPreds: Dataset[TestPredictionsDataset]
                 ): Double = {
    // Mean Percentile Rank Score
    import spark.implicits._
    val pctWindow = Window.partitionBy("userId").orderBy(col("prediction").desc)

    val dfTestWithPreds = inDsTest.join(inDsPreds, Seq("userId", "itemId"))

    dfTestWithPreds.select(col("userId"), col("itemId"), col("freq")
      , percent_rank().over(pctWindow).as("percent_rank")
    ).map(r => r.getInt(2) * r.getDouble(3)*100).reduce(_+_) /
      inDsTest.map(_.freq).reduce(_+_)
  }

  def main(args: Array[String]) {
    val spark : SparkSession = SparkSession
      .builder()
      .master("local")
      .appName("Item-Rec-Engine")
      .config("spark.rdd.compress", true)
      .config("spark.executor.memory", "8g")
      .config("spark.driver.maxResultSize", 0)
      .getOrCreate()

    val inDir = "/Users/deysi/Downloads/instacart_2017_05_01/"

    val mapFiletypeToFilenames: Map[String, String] = Map(
      "orders" -> "orders.csv",
      "priors" -> "order_products__prior.csv"
    )

    import spark.implicits._
    val dsOrders: Dataset[OrdersDataset] = getRawData(spark, "orders", inDir, mapFiletypeToFilenames)
      .as[OrdersDataset]
      .filter(_.eval_set=="prior")

    val dsPriorOrders: Dataset[PriorOrdersDataset] = getRawData(spark, "priors", inDir, mapFiletypeToFilenames)
      .as[PriorOrdersDataset]

    val dsUserItemFreq: Dataset[UserItemFreqDataset] = getUserItemFreqDataset(spark, dsPriorOrders, dsOrders)
    dsUserItemFreq.repartition(4).cache()

    val (dsTrain, dsTest) : (Dataset[UserItemFreqDataset], Dataset[UserItemFreqDataset]) =
      getTrainTestDatasets(spark, dsUserItemFreq)
    println(s"***** DONE Getting Train + Test datasets *****")
    // dsUserItemFreq has (rows, users, items) -> (13307953,206209,49677)
    // dsTrain has (rows, users, items) -> (9284926,206209,49677)
    // dsTest has (rows, users, items) -> (209995705,206209,49677)

    dsTest.repartition(4).cache()
    dsTest.count()
    println(s"***** DONE Caching Test dataset *****")

    val dsTestForPredict: Dataset[TestDatasetForPrediction] = dsTest.drop(col("freq"))
      .toDF("userId", "itemId").as[TestDatasetForPrediction]

    val paramsForALSImplicit: ALSImplicitParams = ALSImplicitParams(
      nonNegativeFlag = true
      , coldStartStrategy = "drop"
      , numLatentFactors = 140
      , maxIters = 10
      , alpha = 40D
      , lambdaForReg = 0.01D
    )

    val (model, dsPreds) : (ALSModel, Dataset[TestPredictionsDataset]) =
      runALSToGetPredictions(spark, dsTrain, dsTestForPredict, paramsForALSImplicit)
    println(s"***** ALS Implicit ran with params: ${paramsForALSImplicit.toString} *****")
    dsPreds.repartition(4).cache()
    println(s"***** DONE Caching Predictions dataset *****")

    val mpr: Double = getMPRScore(spark, dsTest, dsPreds)
    println(s"***** ALS implementation has MPR score of ${mpr} *****")

    val userRecs: DataFrame = model.recommendForAllItems(10)
    println(s"***** DONE Recommending Products For All Users *****")

    spark.stop()
  }

  case class OrdersDataset(
                            order_id: Int
                            , user_id: Int
                            , eval_set: String
                            , order_number: Int
                            , order_dow: Int
                            , order_hour_of_day: Int
                            , days_since_prior_order: Option[Double]
                          )

  case class PriorOrdersDataset(
                                 order_id: Int
                                 , product_id: Int
                                 , add_to_cart_order: Int
                                 , reordered: Int
                               )

  case class UserItemDataset(
                              userId: Int
                              , itemId: Int
                            )

  case class UserItemFreqDataset(
                                  userId: Int
                                  , itemId: Int
                                  , freq: Int
                                )

  case class TestDatasetForPrediction(
                                       userId: Int
                                       , itemId: Int
                                     )

  case class ALSImplicitParams(
                                nonNegativeFlag: Boolean
                                , coldStartStrategy: String
                                , numLatentFactors : Int
                                , maxIters : Int
                                , alpha : Double
                                , lambdaForReg : Double
                              )

  case class TestPredictionsDataset(
                                     userId: Int
                                     , itemId: Int
                                     , prediction: Float
                                   )

}
