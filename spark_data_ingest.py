# Install packages
# noinspection PyUnresolvedReferences
import pandas as pd
# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
import matplotlib as mp
# noinspection PyUnresolvedReferences
from pyspark.sql import *
# noinspection PyUnresolvedReferences
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# Create Spark DataFrame & display first 10 rows
tweet_raw = (spark.read.format("csv").options(header="true").load("20200312_Coronavirus_Tweets_Subset.CSV"))
display(tweet_raw)
tweet_raw.show(10, False)

# Initially filter tweets in English
# noinspection PyUnresolvedReferences
from pyspark.sql.functions import explode

tweet_filter = tweet_raw.select(tweet_raw['account_lang'] == 'en')
tweet_filter.show(10, False)


