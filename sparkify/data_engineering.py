#!/usr/bin/env python3

##
# data_engineering.py
##

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, lit, col
from pyspark.sql.types import IntegerType, FloatType

# Variable to control the size of the timebins that we'll use
# Since subscriptions are likely monthly based, I'm going to look at activity in a two week period
# And try to predict if there will be churn in the next two week bin
# Also this will help to normalize/smooth out irregularities that might arise from day to day usage variations (wkends)
# Note: we can always come back and re-adjust to be something more granular like daily or weekly bins
TIMEBIN_DAYS = 14

##
# Load the dataset file
def load_file(spark, filepath='mini_sparkify_event_data.json'):
    '''
    Load the dataset file
    
    INPUT:
        - spark session
        - filepath to load
    
    OUTPUT:
        - loaded spark dataframe
    '''
    df = spark.read.json(filepath)
    return df

##
# Clean the df
def clean_data(df):
    '''
    Clean the data
    
    INPUT:
        - dataframe to clean
        
    OUTPUT:
        - cleaned dataframe
    '''
    # Require a valid userId
    clean_df = df.dropna(how = "any", subset = ["userId"])
    clean_df = clean_df.filter(clean_df["userId"] != "")
    return clean_df

##
# Feature Engineering
def feature_engineering(spark, df):
    '''
    Feature Engineering
    
    INPUT:
        - spark session
        - dataframe
        
    OUTPUT:
        - dataframe with engineered features
    '''
    
    # DaysSinceRegistration
    def fe_DaysSinceRegistration(df):
        days_active = udf(lambda x,y: float((x-y)/(86400000.0)), FloatType())
        df = df.withColumn("DaysSinceRegistration", days_active("ts", "registration"))
        return df

    # UserTimeBin
    def fe_UserTimeBin(df, days):
        user_timebin = udf(lambda x: int(x/days), IntegerType())
        df = df.withColumn("UserTimeBin", user_timebin("DaysSinceRegistration"))
        return df

    # Initialize UserTimeBin Features dataframe
    def fe_InitializeFeaturesDF(spark, df):
        query='''
            SELECT UserId, UserTimeBin,
                MAX(IF(gender="M", 1, 0)) AS Gender,
                MAX(IF(level="paid", 1, 0)) AS PaidInTimeBin,
                MAX(IF(level="free", 1, 0)) AS FreeInTimeBin,
                MAX(IF(page="Cancellation Confirmation", 1, 0)) AS ChurnedInTimeBin,
                MAX(IF(page="Submit Downgrade", 1, 0)) AS DowngradedInTimeBin,
                MAX(IF(page="Submit Upgrade", 1, 0)) AS UpgradedInTimeBin,
                MIN(DaysSinceRegistration) AS DaysRegisteredAtTimeBin
            FROM dfView GROUP BY UserId, UserTimeBin
        '''
        resulting_df = spark.sql(query)
        return resulting_df

    # fe_WillChurnInNextBin -- 1 if user will cancel in next timebin
    def fe_WillChurnInNextBin(spark, fe_df):
        query = '''
            SELECT resultingView.UserId, resultingView.UserTimeBin FROM
                resultingView,
                (SELECT UserId, UserTimeBin FROM resultingView
                    WHERE ChurnedInTimeBin=1) AS churnedUserBins
            WHERE
                resultingView.UserId=churnedUserBins.UserId
                AND resultingView.UserTimeBin=(churnedUserBins.UserTimeBin-1)
        '''
        resulting_df = spark.sql(query)
        resulting_df = resulting_df.withColumn("WillChurnInNextBin", lit(1))
        # Join to fe_df on ("UserId","UserTimeBin")
        fe_df = fe_df.join(resulting_df, ["UserId","UserTimeBin"], 'left')
        return fe_df

    # fe_PreviouslyDowngraded -- 1 if user downgraded in a previous timebin
    def fe_PreviouslyDowngraded(spark, fe_df):
        query = '''
            SELECT resultingView.UserId, resultingView.UserTimeBin FROM
                resultingView,
                (SELECT UserId, UserTimeBin FROM resultingView
                    WHERE DowngradedInTimeBin=1) AS downgradedUserBins
            WHERE
                resultingView.UserId=downgradedUserBins.UserId
                AND resultingView.UserTimeBin>downgradedUserBins.UserTimeBin
        '''
        resulting_df = spark.sql(query)
        resulting_df = resulting_df.withColumn("PreviouslyDowngraded", lit(1))
        # Join to fe_df on ("UserId","UserTimeBin")
        fe_df = fe_df.join(resulting_df, ["UserId","UserTimeBin"], 'left')
        return fe_df

    # fe_PageCount - encode count of visits per page
    def fe_PageCount(df, fe_df):
        page_df = df.groupBy("UserId","UserTimeBin").pivot("page").count()
        fe_df = fe_df.join(page_df, ["UserId", "UserTimeBin"], 'inner')
        return fe_df

    # fe_SessionsInTimeBin - distinct session count within the timebin
    def fe_SessionsInTimeBin(spark, fe_df):
        query = '''
            SELECT UserId, UserTimeBin, COUNT(DISTINCT(sessionId)) as SessionsInTimeBin
            FROM dfView GROUP BY UserId, UserTimeBin
        '''
        resulting_df = spark.sql(query)
        fe_df = fe_df.join(resulting_df, ["UserId","UserTimeBin"], 'inner')
        return fe_df

    # fe_ArtistsInTimeBin - distinct artist count within the timebin
    def fe_ArtistsInTimeBin(spark, fe_df):
        query = '''
            SELECT UserId, UserTimeBin, COUNT(DISTINCT(artist)) as ArtistsInTimeBin
            FROM dfView GROUP BY UserId, UserTimeBin
        '''
        resulting_df = spark.sql(query)
        fe_df = fe_df.join(resulting_df, ["UserId","UserTimeBin"], 'inner')
        return fe_df

    # fe_DistinctSongsInTimeBin - distinct song count within the timebin
    def fe_DistinctSongsInTimeBin(spark, fe_df):
        query = '''
            SELECT UserId, UserTimeBin, COUNT(DISTINCT(song)) as DistinctSongsInTimeBin
            FROM dfView GROUP BY UserId, UserTimeBin
        '''
        resulting_df = spark.sql(query)
        fe_df = fe_df.join(resulting_df, ["UserId","UserTimeBin"], 'inner')
        return fe_df

    # WillChurnSoon - 1 if will churn in current bin or next
    def fe_WillChurnSoon(spark, fe_df):
        fe_df.createOrReplaceTempView("resultingView")
        query='''
            SELECT UserId, UserTimeBin,
                MAX(IF(ChurnedInTimeBin=1 OR WillChurnInNextBin=1, 1, 0)) AS WillChurnSoon
            FROM resultingView GROUP BY UserId, UserTimeBin
        '''
        resulting_df = spark.sql(query)
        fe_df = fe_df.join(resulting_df, ["UserId","UserTimeBin"], 'inner')
        return fe_df    
    
    ##
    # Main of feature_engineering()

    # Add DaysSinceRegistration
    df = fe_DaysSinceRegistration(df)

    # Create UserTimeBins in number of days
    df = fe_UserTimeBin(df, TIMEBIN_DAYS)
    df.createOrReplaceTempView("dfView")

    # Initialize UserTimeBin features df for future modeling against
    fe_df = fe_InitializeFeaturesDF(spark, df)
    fe_df.createOrReplaceTempView("resultingView")

    # Add WillChurnInNextBin (a binary label we'll try to predict)
    fe_df = fe_WillChurnInNextBin(spark, fe_df)

    # Add PreviouslyDowngraded binary feature
    fe_df = fe_PreviouslyDowngraded(spark, fe_df)

    # Add count per page feature
    fe_df = fe_PageCount(df, fe_df)

    # Add distinct count features
    fe_df = fe_SessionsInTimeBin(spark, fe_df)
    fe_df = fe_ArtistsInTimeBin(spark, fe_df)
    fe_df = fe_DistinctSongsInTimeBin(spark, fe_df)

    # Add WillChurnSoon label
    fe_df = fe_WillChurnSoon(spark, fe_df)
    
    # Replace nulls with 0s
    fe_df = fe_df.na.fill(0)

    # Convert all features/label values to be integers
    fe_df = fe_df.select([col(c).cast("integer") for c in fe_df.columns])

    # Remove temporary views
    spark.catalog.dropTempView("dfView")
    spark.catalog.dropTempView("resultingView")

    # Return feature engineered dataframe
    return fe_df

##
# Display DF Info
def displayDFinfo(df, displayN=10):
    '''
    Display dataframe info
    
    INPUT:
        - dataframe
        - number of records to show
    '''
    print("\nDisplaying Dataframe Info:\n")
    print("DF Schema:\n")
    df.printSchema()
    print("DF Count:\n", df.count())
    print("DF Distributions:\n")
    df.describe().show()
    print("DF Sample:\n")
    df.show(n=displayN)
    print("\nDone.\n")

##
# Main
def main():
    # Create a Spark session
    spark = SparkSession.builder.appName("Sparkify").getOrCreate()
    spark.catalog.clearCache()

    # Load the data from file
    df = load_file(spark)

    # Clean the data
    df = clean_data(df)

    # Engineer features
    fe_df = feature_engineering(spark, df)

    # Display the info about the final dataframe
    #displayDFinfo(fe_df)
    
    return fe_df

if __name__ == '__main__':
    main()
