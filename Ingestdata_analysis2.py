# -*- coding: utf-8 -*-
"""Covid_Nay.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18LxCJmobliA5EbbjGVzH0BagQbJOg2fp
"""

pip install pyspark

from google.colab import drive
drive.mount('/content/drive')

# Install packages
# noinspection PyUnresolvedReferences
import matplotlib as mp
# noinspection PyUnresolvedReferences
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from pyspark.sql import *
# noinspection PyUnresolvedReferences
from pyspark.sql import SparkSession
# noinspection PyUnresolvedReferences
from pyspark.sql.functions import explode
# noinspection PyUnresolvedReferences
from pyspark.sql import functions as F
# noinspection PyUnresolvedReferences
from pyspark.sql.functions import *
# noinspection PyUnresolvedReferences
from pyspark.sql.functions import to_timestamp
# noinspection PyUnresolvedReferences
from pyspark.sql.types import IntegerType
# noinspection PyUnresolvedReferences
from pyspark.sql.functions import hour, mean, count, minute, second
from pyspark.sql import SparkSession
from pyspark.sql import Row

!java -version
!sudo update-alternatives --config java
!java -version
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
!update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
!java -version

!ls /usr/lib/jvm
# Installing JVM
!apt-get install openjdk-8-jdk-headless -qq > /dev/null

# Installing spark from spark website. 
!wget -q https://downloads.apache.org/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz

# Unzipping 
!tar xf spark-2.4.5-bin-hadoop2.7.tgz

# Let us allow to find the spark & set the path variable 
!pip install -q findspark

import os       #importing os to set environment variable
def install_java():
  !apt-get install -y openjdk-8-jdk-headless -qq > /dev/null      #install openjdk
  os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"     #set environment variable
  !java -version       #check java version
install_java()

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.3.2-bin-hadoop2.7"

tweet_raw = (spark.read.format("csv").options(header="true").load("20200312_Coronavirus_Tweets_Subset.CSV"))
display(tweet_raw)
tweet_raw.show(10, False)

#### CLEAN DATA ####
# Initially filter tweets in English & create new filtered DataFrame
tweet_filter = tweet_raw.select("*", F.when(tweet_raw.lang == 'en', 'TRUE').alias('eng_true'))
tweet_filter = tweet_filter.filter("eng_true == 'TRUE'")
# Fix Date Structure
tweet_filter = tweet_filter.withColumn('created_at', regexp_replace('created_at', 'T', ' '))
tweet_filter = tweet_filter.withColumn('created_at', regexp_replace('created_at', 'Z', ''))
# Convert to Timestamp
tweet_filter = tweet_filter.withColumn('dt',to_timestamp(tweet_filter.created_at, 'yyyy-MM-dd HH:mm:ss'))
# Drop Unused Columns
tweet_filter = tweet_filter.drop('created_at','reply_to_status_id','reply_to_user_id','reply_to_screen_name','place_type','account_lang')
# Define Columns for Integer Transformation
cols = spark.createDataFrame([('status_id',1),('user_id',2),('favourites_count',3),('retweet_count',4),('followers_count',5),('friends_count',6)])
cols_col = cols.select("_1")

tweet_filter.show(10, False)

tweet_raw.printSchema()

type(tweet_raw)

# create a spark session
spark = SparkSession.builder\
                    .master("local")\
                    .appName("Structured Streaming")\
                    .getOrCreate()

# Import packages for visualization

import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import io

#Drop NA/Null values in country code column
tweet_raw2 = tweet_filter.na.drop(subset=["country_code"])

from pyspark.sql.functions import col
tweet_raw2.groupBy("country_code") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

# Plot the histogram of langauge that related to Coronavirus
tweet_raw2.groupby(
  'Lang'
).count().select(
  'count'
).rdd.flatMap(
  lambda x: x
).histogram(10)

# Count number of verified accounts
from pyspark.sql.functions import col
tweet_raw2.groupBy("verified") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

#Drop NA/Null values in langauge column
tweet_raw3 = tweet_filter.na.drop(subset=["source"])

from pyspark.sql.functions import col
tweet_raw3.groupBy("source") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

# Plot the histogram of verified accounts
tweet_raw.groupby(
  'source'
).count().select(
  'count'
).rdd.flatMap(
  lambda x: x
).histogram(10)

# Top10 Languages used by Tweets

import io

spark = SparkSession \
    .builder \
    .appName("Twitter Data Analysis") \
    .getOrCreate()
df = tweet_filter
df.createOrReplaceTempView("BtsCovSpo")
sqlDF = spark.sql("select count(*) as Total_count, lang as Language from BtsCovSpo group by lang order by Total_count desc limit 11")
pd = sqlDF.toPandas()


def plot12():
    #plt.title('Top10 Languages used by Tweets')
    #sns.pointplot(y="Total_count", x="Language",data=pd)
    pd.plot.line(x="Language", y="Total_count", title="Top10 Languages used by Tweets")
    bytes_image = io.BytesIO()
    # plt.savefig('foo.png')
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image



# Time series analysis

tweets_pdf = tweet_filter.toPandas()
tweets_pdf.head(10)

import pyspark.sql.functions as f
from pyspark.sql.functions import date_format
tweet_filter = tweet_filter.withColumn('month',f.month(f.to_timestamp('dt', 'dd/MM/yyyy')))

tweet_filter.show()

# Count number of tweets in each month

tweet_filter.groupBy('month').count().show()

tweet_filter2 = tweet_filter.select('dt', date_format('dt', 'u').alias('dow_number'), date_format('dt', 'E').alias('dow_string'))
tweet_filter2.show()

# Count number of tweets each day of the week

tweet_filter2.groupBy('dow_string').count().show()

# Count number of retweeted in each months

from pyspark.sql.types import IntegerType
tweet_filter = tweet_filter.withColumn("retweet_count", tweet_filter["retweet_count"].cast(IntegerType()))
tweet_filter.groupBy('month').sum('retweet_count').show()

import datetime 
import matplotlib.dates as md
import pandas as pd
import numpy as np

# create a spark session
spark = SparkSession.builder\
                    .master("local")\
                    .appName("Structured Streaming")\
                    .getOrCreate()

pd.to_datetime(tweets_pdf['dt'],errors='coerce')
idx = pd.DatetimeIndex(pd.to_datetime(tweets_pdf['dt'],errors='coerce'))

idx

ones = np.ones(len(tweets_pdf['dt']))
ITAvWAL = pd.Series(ones, index=idx)
per_minute = ITAvWAL.resample('1Min').sum().fillna(0)

per_minute

# Commented out IPython magic to ensure Python compatibility.
# The following code is to manipulate the data, and then visualize it into a time series line chart using Matplotlib. 
import sys
import pickle 
# Plotting the series
# %matplotlib inline
fig, ax = plt.subplots()
ax.grid(True)
ax.set_title("Tweet Numbers")
interval = md.MinuteLocator(interval=100)
date_formatter = md.DateFormatter('%H:%M')

#Change number according to the data
datemin = dt.datetime(2020, 3, 12, 00, 00) 
datemax = dt.datetime(2020, 3, 12, 00, 15)


ax.xaxis.set_major_locator(interval) 
ax.xaxis.set_major_formatter(date_formatter) 
ax.set_xlim(datemin, datemax)
max_freq = per_minute.max()
min_freq = per_minute.min()
ax.set_ylim(min_freq-100, max_freq+100) 
ax.plot(per_minute.index, per_minute)
display(fig)



# Sum of confirmed cases around the world
coronadf = (spark.read.format("csv").options(header="true").load("train.csv"))
coronadf.show()

coronadf.groupBy("Country_Region").agg({'ConfirmedCases': 'sum'}).orderBy("sum(ConfirmedCases)", ascending = False).show()

coronadf.groupBy("Province_State").agg({'ConfirmedCases': 'sum'}).orderBy("sum(ConfirmedCases)", ascending = False).show()

coronadf.groupBy("Country_Region").agg({'Fatalities': 'sum'}).orderBy("sum(Fatalities)", ascending = False).show()

# Plot the histogram of Fatalities case in each countries
coronadf.groupby(
  'Country_Region'
).count().select(
  'count'
).rdd.flatMap(
  lambda x: x
).histogram(10)

# Filtering to find only US and ordering by confirmed cases
coronadf2 = coronadf.filter("Country_Region == 'US'").groupBy("Province_State").agg({'ConfirmedCases': 'sum'}).orderBy("sum(ConfirmedCases)", ascending = False)
coronadf.filter("Country_Region == 'US'").groupBy("Province_State").agg({'ConfirmedCases': 'sum'}).orderBy("sum(ConfirmedCases)", ascending = False).show()

from pyspark.sql.window import Window
coronadf2 = coronadf2.withColumn('ConfirmedCases%', f.col('sum(ConfirmedCases)')/f.sum('sum(ConfirmedCases)').over(Window.partitionBy()))
coronadf2.orderBy('ConfirmedCases%', ascending=False).show()

# # Filtering to find only Fatalities rate in the US and ordering by Fatalities
coronadf3 = coronadf.filter("Country_Region == 'US'").groupBy("Province_State").agg({'Fatalities': 'sum'}).orderBy("sum(Fatalities)", ascending = False)
coronadf.filter("Country_Region == 'US'").groupBy("Province_State").agg({'Fatalities': 'sum'}).orderBy("sum(Fatalities)", ascending = False)
coronadf3.withColumn('Fatalities%', f.col('sum(Fatalities)')/f.sum('sum(Fatalities)').over(Window.partitionBy())).show()

