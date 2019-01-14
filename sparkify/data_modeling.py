#!/usr/bin/env python3

##
# data_modeling.py
#
#  Some references used when building this:
#  https://docs.databricks.com/spark/latest/mllib/binary-classification-mllib-pipelines.html
#  https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa
##

import data_engineering as de
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

NUM_FOLDS = 2    # Change to 5 when moving to "production"
NUM_PARALLEL = 4 # Number of parallel models to train

##
# Define Features and Label
def define_features_and_label(df, label_name):
    '''
    Define features and labels
    
    INPUT:
        - df: dataframe to defines features and label against
        - label_name: the name of the column in the df that will be our label
        
    OUTPUT:
        - resulting_df: dataframe with features and label defined
    '''
    
    # Transform all features into a vector using VectorAssembler
    assemblerInputs = ["UserTimeBin", "Gender", "PaidInTimeBin", "FreeInTimeBin", "DowngradedInTimeBin", "UpgradedInTimeBin", 
            "PreviouslyDowngraded", "About", "Add Friend", "Add to Playlist", "Downgrade", "Error", "Help", "Home", "Logout", 
            "NextSong", "Roll Advert", "Save Settings", "Settings", "Thumbs Down", "Thumbs Up", "Upgrade", "SessionsInTimeBin", 
            "ArtistsInTimeBin", "DistinctSongsInTimeBin"]
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    resulting_df = assembler.transform(df)
    
    # Rename label_name column to "label"
    resulting_df = resulting_df.withColumnRenamed(label_name, "label")
    
    return resulting_df



##
# Logistic Regression Model
def lr_model(train, test, evaluator):
    '''
    Logistic Regression Model
    
    INPUT:
        - train dataset for training model
        - test dataset for CV
        - evaluator for model
        
    OUTPUT:
        - best resulting model
    '''
    
    print("Building Logistic Regression Model...")
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label')

    #print(lr.explainParams())
    
    # Create ParamGrid for Cross Validation tuning
    paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.0, 0.1, 0.3])
             .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.3])
             .addGrid(lr.maxIter, [10])
             .build())

    # Create 5-fold CrossValidator
    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid,
                        evaluator=evaluator, numFolds=NUM_FOLDS, parallelism=NUM_PARALLEL)

    # Run cross validations
    cvModel = cv.fit(train)

    # Use test set to measure the accuracy of our model on new data
    predictions = cvModel.transform(test)
    print('[*] Best LR Model AUC PR: ' + str(evaluator.evaluate(predictions)))

    # Print best model info
    print('[*] Best LR Model Intercept: ', cvModel.bestModel.intercept)
    print('[*] Best LR Model Param (maxIter): ', cvModel.bestModel._java_obj.getMaxIter())
    print('[*] Best LR Model Param (regParam): ', cvModel.bestModel._java_obj.getRegParam())
    print('[*] Best LR Model Param (elasticNetParam): ', cvModel.bestModel._java_obj.getElasticNetParam())
    print('[*] Best LR Model Weights:')
    weights = cvModel.bestModel.coefficients
    weights = [(float(w),) for w in weights]
    display(weights)
    print("Done.")
    
    return cvModel.bestModel

##
# Random Forest Model
def rf_model(train, test, evaluator):
    '''
    Random Forest Model
    
    INPUT:
        - train dataset for training model
        - test dataset for CV
        - evaluator for model
        
    OUTPUT:
        - best resulting model
    '''
    
    print("Building Random Forest Model...")
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")

    #print(rf.explainParams())
    
    # Create ParamGrid for Cross Validation tuning
    paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 4, 6])
             .addGrid(rf.maxBins, [12, 24, 32])
             .addGrid(rf.numTrees, [10])
             .build())

    # Create 5-fold CrossValidator
    cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid,
                        evaluator=evaluator, numFolds=NUM_FOLDS, parallelism=NUM_PARALLEL)

    # Run cross validations
    cvModel = cv.fit(train)

    # Use test set to measure the accuracy of our model on new data
    predictions = cvModel.transform(test)
    print('[*] Best RF Model AUC PR: ' + str(evaluator.evaluate(predictions)))

    # Print best model info
    print('[*] Best RF Model Param (maxDepth): ', cvModel.bestModel._java_obj.getMaxDepth())
    print('[*] Best RF Model Param (maxBins): ', cvModel.bestModel._java_obj.getMaxBins())
    print('[*] Best RF Model Param (numTrees): ', cvModel.bestModel._java_obj.getNumTrees())
    print("Done.")
    
    return cvModel.bestModel

##
# Gradient-Boosted Tree Classifier
def gbt_model(train, test, evaluator):
    '''
    Gradient-Boosted Tree Model
    
    INPUT:
        - train dataset for training model
        - test dataset for CV
        - evaluator for model
        
    OUTPUT:
        - best resulting model
    '''
    
    print("Building Gradient-Boosted Tree Model...")
    gbt = GBTClassifier(labelCol="label", featuresCol="features")

    #print(gbt.explainParams())
    
    paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [3, 5, 8])
             .addGrid(gbt.maxBins, [4, 12, 24])
             .addGrid(gbt.maxIter, [10])
             .build())

    # Create 5-fold CrossValidator
    cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid,
                        evaluator=evaluator, numFolds=NUM_FOLDS, parallelism=NUM_PARALLEL)

    # Run cross validations
    cvModel = cv.fit(train)

    # Use test set to measure the accuracy of our model on new data
    predictions = cvModel.transform(test)
    print('[*] Best GBT Model AUC PR: ' + str(evaluator.evaluate(predictions)))

    # Print best model info
    print('[*] Best GBT Model Param (maxDepth): ', cvModel.bestModel._java_obj.getMaxDepth())
    print('[*] Best GBT Model Param (maxBins): ', cvModel.bestModel._java_obj.getMaxBins())
    print('[*] Best GBT Model Param (maxIter): ', cvModel.bestModel._java_obj.getMaxIter())
    print("Done.")
    
    return cvModel.bestModel


##
# Main
def main():
    
    # Get data from data_engineering step
    df = de.main()
    df.cache()
    
    # Class imbalance sample rates
    sample_fractions = {
        "ChurnedInTimeBin" : 0.065,
        "WillChurnInNextBin" : 0.05,
        "WillChurnSoon" : 0.12
    }
    
    # Use areaUnderPR as evaluation metric
    evaluator = BinaryClassificationEvaluator(metricName = 'areaUnderPR')
    
    # Loop through the labels we're going to try and build models for
    for label, sample_fraction in sample_fractions.items():
        
        print("Building model for predicting this label: ", label)
        
        # Define features and labels into a dataframe for modeling
        dataset = define_features_and_label(df, label) 
        
        # Sample the data b/c of the class imbalance
        sampled_data = dataset.sampleBy('label', fractions={0: sample_fraction, 1: 1.0})
        dataset.unpersist()
        sampled_data.unpersist()
    
        # Do 80/20 split of the dataset for training/testing
        train, test = sampled_data.randomSplit([0.8, 0.2], seed=147309)
        train.cache()
        test.cache()

        # Train, CrossValidate, and Save models
        # Logistic Regression
        bestLR = lr_model(train, test, evaluator)
        bestLR.write().overwrite().save(label + ".LRmodel")
        del bestLR
        
        # Random Forest
        bestRF = rf_model(train, test, evaluator)
        bestRF.write().overwrite().save(label + ".RFmodel")
        del bestRF
        
        # Gradient-Boosted Trees
        bestGBT = gbt_model(train, test, evaluator)
        bestGBT.write().overwrite().save(label + ".GBTmodel")
        del bestGBT
        
        train.unpersist()
        test.unpersist()
    df.unpersist()


if __name__ == '__main__':
    main()
