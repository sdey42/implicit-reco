package com.roku.recengine

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.collection.immutable
import scala.util.Random

/**
  * Created by deysi on 9/7/17.
  */
object RecEngine {

  def GetRawData(
                  spark: SparkSession,
                  fType: String,
                  inDir: String,
                  mapFtypeToFilenames: Map[String, String]
                ) = {

    spark.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ",")
      .load(inDir + mapFtypeToFilenames(fType))

  }

  def GetUserItemFreqDataset(
                              spark: SparkSession,
                              inDsPrior: Dataset[PriorOrdersDataset],
                              inDsOrders: Dataset[OrdersDataset]
                            ) = {
    // Need to join inDsOrders with inDsPrior on order_id
    import spark.implicits._
    val tmpDsUserItem = inDsPrior.joinWith(inDsOrders, inDsPrior.col("order_id") === inDsOrders.col("order_id"))
      .map(r => UserItemDataset(r._2.user_id, r._1.product_id))
      .as[UserItemDataset]

    tmpDsUserItem
      .map(x => (x.userId, x.itemId, 1))
      .toDF().rdd
      .map(r => ((r.getInt(0), r.getInt(1)) -> r.getInt(2)))
      .reduceByKey((a, b) => (a + b))
      .map(r => UserItemFreqDataset(r._1._1, r._1._2, Some(r._2)))
      .toDS()
      .as[UserItemFreqDataset]
  }

  def VectorizeUserOrders(
                           spark: SparkSession,
                           inDs: Dataset[UserItemFreqDataset],
                           inVecRefs: Option[ReferencesForVectorizedDS]
                         ) = {
    // There are 50k products (items), and 206k (206209) users.
    // Expected output should be a Dataset with 206k rows, with 2 columns, and second column should
    // be Vector with as many elements as number of products (50k), with value being the frequency
    import spark.implicits._
    val (arrUsers, arrItems) =
      if (inVecRefs.isEmpty)
        (inDs.map(_.userId).distinct().collect(), inDs.map(_.itemId).distinct().collect())
      else
        (inVecRefs.get.mapUidToIdx.keys.toArray, inVecRefs.get.mapItemToIdx.keys.toArray)

    val (numUsers, numItems) =
      if (inVecRefs.isEmpty)
        (arrUsers.length, arrItems.length)
      else
        (inVecRefs.get.mapUidToIdx.keys.size, inVecRefs.get.mapItemToIdx.keys.size)

    val (mapIdxToUid, mapIdxToItem) =
      if (inVecRefs.isEmpty)
        (arrUsers.zipWithIndex.map(_.swap).toMap, arrItems.zipWithIndex.map(_.swap).toMap)
      else
        (inVecRefs.get.mapIdxToUid, inVecRefs.get.mapIdxToItem)

    val (mapUidToIdx, mapItemToIdx) =
      if (inVecRefs.isEmpty)
        (mapIdxToUid.map(_.swap), mapIdxToItem.map(_.swap))
      else
        (inVecRefs.get.mapUidToIdx, inVecRefs.get.mapItemToIdx)

    (inDs
      .groupByKey(_.userId)
      .mapGroups((k, v) => (k, {
        val elems = v.map(x => (x.itemId, x.freq))
          .map(x => (mapItemToIdx(x._1), x._2.get.toDouble)).toArray
        Vectors.sparse(numItems, elems)
      })
      ).toDF("userId", "vecOrders")
      .as[UserVectorizedOrders],
      ReferencesForVectorizedDS(
        mapIdxToUid
        , mapUidToIdx
        , mapIdxToItem
        , mapItemToIdx
      ))
  }

  def AddTwoSparseOrderVecs(vec1: SparseVector, vec2: SparseVector): SparseVector = {
    if (vec1.size == vec2.size & vec1.size > 0) {
      val arrIdxs: Set[Int] = vec1.indices.toSet.union(vec2.indices.toSet)
      Vectors.sparse(vec1.size, arrIdxs.map(i => (i, vec1(i) + vec2(i))).toSeq).toSparse
    } else Vectors.dense(Array.empty[Double]).toSparse
  }

  def GetTrainTestDatasets(
                            spark: SparkSession,
                            inDs: Dataset[UserItemFreqDataset],
                            inDsVec: Dataset[UserVectorizedOrders],
                            inVecRefsForInDs: ReferencesForVectorizedDS
                          ) = {
    // 1	make sure that every user appears both in the train and in the test set.
    // Similarly, make sure that every product appears both in the train and in the test set.
    // 2	if a pair <user_x, product_y> appears in the training set at least once, it must not appear in the test set
    // 3	in the test set, for each user, you will also have to add 1000 random products that that particular user has not bought yet.
    val NUM_UNSEEN_PRODUCTS: Int = 1000
    val (numUsers, numItems) = (inVecRefsForInDs.mapUidToIdx.keys.size, inVecRefsForInDs.mapItemToIdx.keys.size)
    println(s"***** Num of (users,items) in input dataset, inDs: ${(numUsers, numItems)} *****") // (206209, 49677)
    println(s"***** Checkpoint 0 *****")

    // Guaranteed to not have same (uid, item) in TRAIN + TEST
    val arrRawDsTrainDsTest: Array[Dataset[UserItemFreqDataset]] = inDs
      .randomSplit(Array(0.7D, 0.3D), seed = System.currentTimeMillis())

    println(s"***** Checkpoint 1 *****")

    // START - TEST Set prep
    // Fulfilling the condition of introducing 1000 new (unseen) items per user into TEST Set
    // This takes care of TEST Set having every user, but there could be missed items
    import spark.implicits._
    val dsTestNewItemsForUsers: Dataset[UserVectorizedOrders] = {
      inDsVec.map(r => {
        (r.userId,
          Vectors.sparse(numItems, {
            Random.shuffle(
              (0 to numItems - 1) diff r.vecOrders.toSparse.indices
            )
              .take(NUM_UNSEEN_PRODUCTS)
              .map(x => (x, -1D))
          }))
      })
        .toDF("userId", "vecOrders")
        .as[UserVectorizedOrders]
    }

    println(s"***** Checkpoint 2 *****")

    val arrVecAndRefObjForRawDsTrainDsTest = arrRawDsTrainDsTest.map(VectorizeUserOrders(spark, _, Some(inVecRefsForInDs)))

    val dsTestWithNewItems: Dataset[UserVectorizedOrders] = arrVecAndRefObjForRawDsTrainDsTest(1)._1
      .union(dsTestNewItemsForUsers)
      .groupByKey(_.userId)
      .reduceGroups((x, y) =>
        UserVectorizedOrders(x.userId, AddTwoSparseOrderVecs(x.vecOrders.toSparse, y.vecOrders.toSparse))
      )
      .map(_._2)
      .toDF("userId", "vecOrders")
      .as[UserVectorizedOrders]

    val tupMissingUsersItemsInTrain: (Option[Seq[Int]], Option[Seq[Int]]) = (
      if (arrVecAndRefObjForRawDsTrainDsTest(0)._2.mapUidToIdx.keys.size < numUsers) {
        Some(
          inVecRefsForInDs.mapUidToIdx.keys.toSeq
            diff arrVecAndRefObjForRawDsTrainDsTest(0)._1.map(_.userId).distinct().collect()
        )
      } else None
      ,
      if (arrVecAndRefObjForRawDsTrainDsTest(0)._2.mapItemToIdx.keys.size < numItems) {
        Some(
          inVecRefsForInDs.mapItemToIdx.keys.toSeq
            diff
            arrVecAndRefObjForRawDsTrainDsTest(0)._1.rdd
              .map(x => {
                x.vecOrders.toSparse.indices.map(inVecRefsForInDs.mapIdxToItem)
              }.toSet
              )
              .reduce((s1, s2) => (s1 union s2))
              .toSeq
        )
      } else None
      )

    println(s"***** Checkpoint 3 *****")

    // Collect those items that are missing in the TEST Set
    val arrMissingItemsInTest: Option[Array[Int]] =
      if (arrVecAndRefObjForRawDsTrainDsTest(1)._2.mapItemToIdx.keys.size == numItems)
        None
      else {
        Some(
          dsTestWithNewItems
            .map(x => x.vecOrders.toArray)
            .reduce((a1, a2) => (a1, a2).zipped.map(_ + _))
            .zipWithIndex
            .filter(_._1 == 0D)
            .map(_._2)
            .map(inVecRefsForInDs.mapIdxToItem(_))
        )
      }

    println(s"***** Checkpoint 4 *****")

    val dsTest: Dataset[UserVectorizedOrders] = if (arrMissingItemsInTest.isDefined) {
      dsTestWithNewItems.union(
        spark.createDataset(
          arrMissingItemsInTest.get.map(itmId => {
            (Random.shuffle(
              inVecRefsForInDs.mapUidToIdx.keys.toList.diff(
                inDs.filter(_.itemId == itmId).map(_.userId).collect())
            ).take(1).apply(0),
              Vectors.sparse(numItems, Seq((inVecRefsForInDs.mapItemToIdx(itmId), Double.MinValue)))
              )
          }))
          .toDF("userId", "vecOrders")
          .as[UserVectorizedOrders]
      )
        .groupByKey(_.userId)
        .reduceGroups((x, y) =>
          UserVectorizedOrders(x.userId, AddTwoSparseOrderVecs(x.vecOrders.toSparse, y.vecOrders.toSparse))
        )
        .map(_._2)
        .toDF("userId", "vecOrders")
        .as[UserVectorizedOrders]
    } else dsTestWithNewItems

    println(s"***** Checkpoint 5 *****")

    // START - TRAIN Set prep

    // We're assured that there are no (uid, item) rows that are in TRAIN, and also in TEST
    // We need to inject (uid, item) pairs for missing items/users if any
    val dsSupplementForTrain: Option[Dataset[UserVectorizedOrders]] =
      if (tupMissingUsersItemsInTrain._1.isDefined | tupMissingUsersItemsInTrain._2.isDefined) {
        val dsTmpTrainMissingUsers = tupMissingUsersItemsInTrain._1.get.toStream.flatMap(
          x => tupMissingUsersItemsInTrain._2.get.toStream.map(y => (x, y))
        )
          .take(math.max(tupMissingUsersItemsInTrain._1.size, tupMissingUsersItemsInTrain._2.size) * 2)
          .toSeq
          .map(r =>
            UserVectorizedOrders(r._1, Vectors.sparse(numItems, Seq((r._2, Double.MinValue))))
          )
          .toDF("userId", "vecOrders")
          .as[UserVectorizedOrders]

        Some(
          dsTmpTrainMissingUsers.joinWith(dsTest, dsTmpTrainMissingUsers.col("userId") === dsTest.col("userId"), "inner")
            .map(x => (x._1.userId, x._1.vecOrders, x._2.vecOrders))
            .as[(Int, Vector, Vector)]
            .map({ case (uid, vTrainNewItms, vTest) =>
              UserVectorizedOrders(uid,
                Vectors.sparse(numItems,
                  vTrainNewItms.toSparse.indices diff vTest.toSparse.indices map { idx => (idx, vTrainNewItms(idx)) }
                )
              )
            })
            .toDF("userId", "vecOrders")
            .as[UserVectorizedOrders]
        )

      }
      else None

    println(s"***** Checkpoint 6 *****")

    val dsTrain: Dataset[UserVectorizedOrders] = if (dsSupplementForTrain.isDefined) {
      arrVecAndRefObjForRawDsTrainDsTest(0)._1 union dsSupplementForTrain.get
        .groupByKey(_.userId)
        .reduceGroups((x, y) =>
          UserVectorizedOrders(x.userId, AddTwoSparseOrderVecs(x.vecOrders.toSparse, y.vecOrders.toSparse))
        )
        .map(_._2)
        .toDF("userId", "vecOrders")
        .as[UserVectorizedOrders]
    } else arrVecAndRefObjForRawDsTrainDsTest(0)._1

    println(s"***** Checkpoint 7 *****")

    (dsTrain, dsTest)
  }

  def NormalizeTrainDataset(
                             spark: SparkSession,
                             inDs: Dataset[UserVectorizedOrders]
                           ): Dataset[MeanNormalizedTrainDataset] = {
    import spark.implicits._
    val scaler = new StandardScaler()
      .setInputCol("vecOrders")
      .setOutputCol("scaledVecOrders")
      .setWithMean(true)
      .setWithStd(true)

    val model = scaler.fit(inDs)

    val outDs = model.transform(inDs).as[MeanNormalizedTrainDataset]
    println(s"***** Num of rows in outDs : ${outDs.count()} *****")
    outDs
  }

  def CreateTrainRatingsDSForALS(
                                  spark: SparkSession,
                                  inDsTrain: Dataset[MeanNormalizedTrainDataset],
                                  inDsVecRefs: ReferencesForVectorizedDS
                                ): Dataset[RatingDataset] = {
    import spark.implicits._
    val rddUidItmOrd: RDD[(Int, Int, Float)] = inDsTrain.flatMap(_.scaledVecOrders.toSparse.indices).rdd
      .zip(
        inDsTrain.flatMap(_.scaledVecOrders.toSparse.values).rdd
      ).zip(
      inDsTrain.flatMap(r => Array.fill(r.scaledVecOrders.toSparse.indices.size)(r.userId)).rdd
    ).map(x => (x._2, inDsVecRefs.mapIdxToItem(x._1._1), x._1._2.toFloat))

    spark.createDataset(rddUidItmOrd).toDF("userId", "itemId", "rating")
      .as[RatingDataset]
  }

  def CreateTestDSForALS(
                          spark: SparkSession,
                          inDsTest: Dataset[UserVectorizedOrders],
                          inDsVecRefs: ReferencesForVectorizedDS
                        ): Dataset[TestDatasetForPrediction] = {
    import spark.implicits._
    val rddUidItm: RDD[(Int, Int)] = inDsTest.flatMap(_.vecOrders.toSparse.indices).rdd
      .zip(
        inDsTest.flatMap(r => Array.fill(r.vecOrders.toSparse.indices.size)(r.userId)).rdd
      ).map(x => (x._2, inDsVecRefs.mapIdxToItem(x._1)))

    spark.createDataset(rddUidItm).toDF("userId", "itemId")
      .as[TestDatasetForPrediction]
  }

  def RunALSToGetPredictions(
                              spark: SparkSession,
                              inDsTrain: Dataset[RatingDataset],
                              inDsTest: Dataset[TestDatasetForPrediction],
                              params: ALSImplicitParams,
                              outModelPath: String): (ALSModel, Dataset[TestPredictionsDataset]) = {
    val als = new ALS()
      .setImplicitPrefs(params.implicitPrefs)
      .setMaxIter(params.maxIters)
      .setRank(params.numLatentFactors)
      .setAlpha(params.alpha)
      .setRegParam(params.lambdaForReg)
      //.setNumBlocks(params.numBlocks)
      .setCheckpointInterval(params.checkpointInterval)
      .setFinalStorageLevel("MEMORY_AND_DISK_SER")
      .setUserCol("userId")
      .setItemCol("itemId")
      .setRatingCol("rating")
      .setPredictionCol("prediction")

    println(s"***** START - Fitting ALS Model to Train *****")
    val model: ALSModel = als.fit(inDsTrain)
    println(s"***** END - Fitting ALS Model to Train *****")

    println(s"***** START - Save out trained ALS Model to ${outModelPath} *****")
    model.save(outModelPath)
    println(s"***** END - Save out ALS Model to ${outModelPath} *****")

    println(s"***** START - Making Test Predictions *****")
    import spark.implicits._
    val predictions: Dataset[TestPredictionsDataset] = model.transform(inDsTest).toDF("userId", "itemId", "prediction")
      .as[TestPredictionsDataset]
    println(s"***** END - Making Test Predictions *****")
    (model, predictions)
  }

  def GetRMSEScore(inPreds: Dataset[TestPredictionsDataset]): Double = {
    // Root Mean Square Error Score
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    println(s"***** START - Evaluating RMSE *****")
    val rmse = evaluator.evaluate(inPreds)
    println(s"***** END - Evaluating RMSE to be ${rmse} *****")
    rmse
  }

  def GetMPRScore(
                   spark: SparkSession,
                   inPreds: Dataset[TestPredictionsDataset],
                   //inDsVecRefs: ReferencesForVectorizedDS
                   inNumItems: Int = 49677
                 ) = {
    // Mean Percentile Rank Score
    import spark.implicits._
    println(s"***** START - Calculating r_ui *****")
    val dsRui: Dataset[(Int, immutable.Seq[(Int, Double, Int)])] = inPreds.groupByKey(_.userId)
      .mapGroups((uid, itms) =>
        (uid, itms.toStream.sortWith(_.prediction >= _.prediction)
          .zipWithIndex
          .map(x => (x._1.itemId, x._1.prediction.toDouble, x._2))
          .toSeq))
    // uid | (itemId,prediction,idx)
    println(s"***** DONE - Calculating r_ui *****")
    println(s"***** START - Calculating r_ui * rank_ui *****")
    val dsRuiTimesRankUI: Dataset[(Int, immutable.Seq[(Int, Double)])] = dsRui
      .map(x =>
        (x._1, x._2
          .map(y =>
            (y._1,
              //(y._2 * (y._3*100/(inDsVecRefs.mapItemToIdx.keySet.size - 1)))
              (y._2 * (y._3*100/(inNumItems)))
              ))
          ))
    println(s"***** DONE - Calculating r_ui * rank_ui *****")
    val nr: Double = dsRui.flatMap(_._2.map(_._2)).reduce(_+_)
    val dr: Double = dsRuiTimesRankUI.flatMap(_._2.map(_._2)).reduce(_+_)
    val mpr: Double = nr/dr
    println(s"DONE - Calculating MPR Score with ${mpr}")
    mpr
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
    val writeToDisk = false
    val loadFromDisk = true
    val samplePct : Double = 0.05D // default 1
    val outModelPath = "/Users/deysi/Downloads/roku/out/model/"
    val trainPathOnDisk = "/Users/deysi/Downloads/roku/out/train/dstrain"
    val testPathOnDisk = "/Users/deysi/Downloads/roku/out/test/dstest"
    val runID = System.currentTimeMillis()

    val mapFiletypeToFilenames: Map[String, String] = Map(
      "orders" -> "orders.csv",
      "priors" -> "order_products__prior.csv"
    )

    val (dsTrainRatings, dsTestForPredict) =
    if (loadFromDisk) {
      import spark.implicits._
      if (samplePct > 0D & samplePct < 1)
        (spark.read.parquet(trainPathOnDisk).sample(false, samplePct).as[RatingDataset],
          spark.read.parquet(testPathOnDisk).sample(false, samplePct).as[TestDatasetForPrediction])
      else
        (spark.read.parquet(trainPathOnDisk).as[RatingDataset],
          spark.read.parquet(testPathOnDisk).as[TestDatasetForPrediction])
    } else {
      import spark.implicits._
      val dsOrders: Dataset[OrdersDataset] = GetRawData(spark, "orders", inDir, mapFiletypeToFilenames)
        .as[OrdersDataset]
        .filter(_.eval_set=="prior")

      val dsPriorOrders: Dataset[PriorOrdersDataset] = GetRawData(spark, "priors", inDir, mapFiletypeToFilenames)
        .as[PriorOrdersDataset]

      println(s"***** dsOrders: ${dsOrders.count()} *****")
      dsOrders.show(5)
      println(s"***** dsPriorOrdersPriors: ${dsPriorOrders.count()} *****")
      dsPriorOrders.show(5)

      val dsUserItemFreq: Dataset[UserItemFreqDataset] = GetUserItemFreqDataset(spark, dsPriorOrders, dsOrders)
      dsUserItemFreq.repartition(4).cache()
      println(s"***** dsUserItemFreq: ${dsUserItemFreq.count()} *****")
      dsUserItemFreq.show(5)

      val objDsUserVecsAndVecReferences = VectorizeUserOrders(spark, dsUserItemFreq, None)
      val dsUserVecs = objDsUserVecsAndVecReferences._1.repartition(4).cache()
      val refsForDsUserVecs = objDsUserVecsAndVecReferences._2
      println(s"***** dsUserVecs: ${dsUserVecs.count()} *****")
      dsUserVecs.show(5)

      val (dsTrain, dsTest) =
        if (samplePct > 0D & samplePct < 1D) {
          GetTrainTestDatasets(spark,
            dsUserItemFreq,
            dsUserVecs.sample(false, samplePct),
            refsForDsUserVecs
          )
        } else {
          GetTrainTestDatasets(spark,
            dsUserItemFreq,
            dsUserVecs,
            refsForDsUserVecs
          )
        }

      println(s"***** Num of (rows, users) in dsTrain: ${(dsTrain.count(), dsTrain.map(_.userId).distinct().count())} *****")
      dsTrain.show(5)

      println(s"***** Num of (rows, users) in dsTest: ${(dsTest.count(), dsTest.map(_.userId).distinct().count())} *****")
      dsTest.show(5)

      val dsTrainMN = NormalizeTrainDataset(spark, dsTrain)
      println(s"***** Num of rows in dsTrainMN : ${dsTrainMN.count()} *****")
      dsTrainMN.show(5)

      (CreateTrainRatingsDSForALS(spark, dsTrainMN, refsForDsUserVecs),
        CreateTestDSForALS(spark, dsTest, refsForDsUserVecs))
    }

    if (writeToDisk) {
      println(s"***** START - Saving dsTrainRatings and dsTestForPredict to Disk *****")
      dsTrainRatings.write.parquet(trainPathOnDisk+runID)
      dsTestForPredict.write.parquet(testPathOnDisk+runID)
      println(s"***** END - Saving dsTrainRatings and dsTestForPredict to Disk *****")
    }

    println(s"***** Num of rows in dsTrainRatings: ${dsTrainRatings.count()} *****")
    dsTrainRatings.show(5)
    println(s"***** Num of rows in dsTestForPredict: ${dsTestForPredict.count()} *****")
    dsTestForPredict.show(5)

    val paramsForALSImplicit = ALSImplicitParams(
      implicitPrefs = true,
      numLatentFactors = 100,
      maxIters = 5,
      alpha = 40D,
      lambdaForReg = 0.01D,
      checkpointInterval = 2
    )

    val (model, predictions) = RunALSToGetPredictions(spark, dsTrainRatings, dsTestForPredict, paramsForALSImplicit, outModelPath)
    println(s"***** For ALS Implicit with the following params: ${paramsForALSImplicit.toString} *****")

    val rmse = GetRMSEScore(predictions)
    val mpr = GetMPRScore(spark, predictions)
    println(s"***** The ALS implementation ended with MPR score of ${mpr} *****")

    spark.stop()
  }

  case class OrdersDataset(
                            order_id: Int,
                            user_id: Int,
                            eval_set: String,
                            order_number: Int,
                            order_dow: Int,
                            order_hour_of_day: Int,
                            days_since_prior_order: Option[Double]
                          )

  case class PriorOrdersDataset(
                                 order_id: Int,
                                 product_id: Int,
                                 add_to_cart_order: Int,
                                 reordered: Int
                               )

  case class UserItemDataset(userId: Int, itemId: Int)

  case class UserItemFreqDataset(userId: Int, itemId: Int, freq: Option[Int])

  case class UserVectorizedOrders(userId: Int, vecOrders: Vector)

  case class ReferencesForVectorizedDS(
                                        mapIdxToUid : Map[Int, Int]
                                        , mapUidToIdx : Map[Int, Int]
                                        , mapIdxToItem : Map[Int, Int]
                                        , mapItemToIdx : Map[Int, Int]
                                      )

  case class MeanNormalizedTrainDataset(userId: Int, vecOrders: Vector, scaledVecOrders: Vector)

  case class RatingDataset(userId: Int, itemId: Int, rating: Float)

  case class TestDatasetForPrediction(userId: Int, itemId: Int)

  case class ALSImplicitParams(
                                implicitPrefs: Boolean,
                                numLatentFactors : Int,
                                maxIters : Int,
                                alpha : Double,
                                lambdaForReg : Double,
                                checkpointInterval: Int
                              )

  case class TestPredictionsDataset(userId: Int, itemId: Int, prediction: Float)

}
