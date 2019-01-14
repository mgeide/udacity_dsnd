# Sparkify Project

## Overview

This project builds models from a streaming music provider's logs in order to predict user "churn" (when a user will cancel their subscription). In this project, three techniques are used to build models: Logistic Regression, Random Forest, and Gradient-Boosted Trees. Seperate models are built to make predictions for three different churn labels that were engineered:

* ChurnedInTimeBin - label meaning that the user cancelled their subscription in the time bin that is being analyzed
* WillChurnInNextBin - label meaning that the user cancels their subscription in the following time bin from the one being analyzed
* WillChurnSoon - label meaning that the user cancels in the time bin or the next (i.e., one of the previous labels is true)

Further explanation of the methodology and details/results from this project were presented in a blogpost here:
https://medium.com/@mike_71681/building-user-churn-models-in-pyspark-3f55e4f49f4

### Overview of Files

Primary Files:
* data_engineering.py - loads the data from mini_sparkify_event_data.json, cleans the data, performs feature engineering, and returns a dataframe
* data_modeling.py - calls data_engineering.py and samples the data due to class imbalance, splits the data, trains, cross-validates, and saves the best model

Notebook Files:
* ExploringData.ipynb - initial notebook that was used to explore the data and work on feature engineering. This is deprecated, as the stable code was moved to data_engineering.py
* SamplingTest.ipynb - this notebook was used to explore data sampling to deal with the class imbalance problem
* RunModeling.ipynb - calls data_modeling.py and displays the output
* lastGBT_run.ipynb - display of last run of GBT model (mem/Spark context errors had previously stopped RunModeling.ipynb at end - adjust JVM memory size accordingly and/or move to Spark cluster)

Data Files:
* mini_sparkify_event_data.json.bz2 - starting data file containing the streaming music provider's user logs (bzip2 -d before use)

Saved Models: saved_models/
* *.LRmodel - saved logistic regression model(s)
* *.RFmodel - saved random forest model(s)
* *.GBTmodel - saved gradient-boosted trees model(s)

Draft of blog post:
* blogpost_draft.pdf

## Instructions

Requirements: python 3 (project was fun on python 3.7.1) and pyspark (project was run on Spark 2.4.0)

* If from Jupyter Notebooks/Lab: open and run the file RunModeling.ipynb
* If from terminal: run data_modeling.py

Be sure to have your data file (mini_sparkify_event_data.json) in the same directory that you are running the script from.

Resulting "model" files will be created in the same directory along with info about each of the models created.

## Future Work

* Explore if it is better to combine into a multi-class predictive model versus multiple binary models. The individual models gave me more control, which is why I used that approach initially.
* Track and include a user’s running average of activity over time to use for feature engineering – maybe a deviation from their average could be a good feature versus just comparing the user’s activity to all users when building a model (i.e., track each user’s pattern of life and deviations from it within time intervals).
* Look at exploring other predictive classification models, such as building a neural net in pytorch; as well as explore dimensionality reduction techniques (e.g., PCA) and resulting latent features.
* Experiment with other sampling techniques (e.g., SMOTE) to deal with the class imbalance (small number of churn events in the log data) and improve the model.
* Make additional classification predictions, such as upgrades and downgrades.
* Create more visual representations of the data and model results for exploration.  
