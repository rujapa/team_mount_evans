# Install packages
# noinspection PyUnresolvedReferences
import pandas as pd
# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
import matplotlib as mp
# noinspection PyUnresolvedReferences
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from pyspark.sql import *
# noinspection PyUnresolvedReferences
from pyspark.sql import SparkSession

# Initialize Spark Session & Context
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# Create Spark DataFrame & display first 10 rows
tweet_raw = (spark.read.format("csv").options(header="true").load("20200312_Coronavirus_Tweets_Subset.CSV"))
display(tweet_raw)
tweet_raw.show(10, False)

# Initially filter tweets in English & create new filtered DataFrame
# noinspection PyUnresolvedReferences
from pyspark.sql.functions import explode
# noinspection PyUnresolvedReferences
from pyspark.sql import functions as F

tweet_filter = tweet_raw.select("*", F.when(tweet_raw.lang == 'en', 'TRUE').alias('eng_true'))
tweet_filter.show()
tweet_filter = tweet_filter.filter("eng_true == 'TRUE'")
tweet_filter.show()

# Create simple tweet sentiment analysis using TextBlob
# noinspection PyUnresolvedReferences
import textblob as tb

positive = 0
negative = 0
neutral = 0
polarity = 0


def percent_calc(a, b):
    return 100 * float(a) / float(b)


# Isolate tweets column for sentiment analysis
tweets_col = tweet_filter.select("text")
tweets = tweets_col.rdd.map(lambda row : row[0]).collect()
n_tweets = len(tweets_col.rdd.map(lambda row : row[0]).collect())

for tweet in tweets:
    print(tweet)
    myAnalysis = tb.TextBlob(tweet)
    polarity += myAnalysis.sentiment.polarity
    if myAnalysis.sentiment.polarity == 0:
        neutral += 1
    elif myAnalysis.sentiment.polarity > 0.00:
        positive += 1
    elif myAnalysis.sentiment.polarity < 0.00:
        negative += 1

positive = percent_calc(positive, n_tweets)
negative = percent_calc(negative, n_tweets)
neutral = percent_calc(neutral, n_tweets)

positive = format(positive, '.2f')
negative = format(negative, '.2f')
neutral = format(neutral, '.2f')

print('----------------------------------------------------------------------------')
if polarity > 0:
    print('Positive')
elif polarity < 0:
    print('Negative')
elif polarity == 0:
    print('Neutral')

# Plot results

labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]','Negative [' + str(negative) + '%]']
sizes = [positive, neutral, negative]
colors = ['green', 'yellow', 'red']
patches, texts = plt.pie(sizes, colors = colors, startangle = 90)
plt.legend(patches, labels, loc = "best")
plt.title('How people are reacting on ' + 'Covid-19' + ' by analyzing ' + str(n_tweets) + ' Tweets.')
plt.axis('equal')
plt.tight_layout()
plt.show()

