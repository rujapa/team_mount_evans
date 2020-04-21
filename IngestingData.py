#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pandas as pd
import pyspark
from pyspark.sql import SparkSession
import numpy as np
import matplotlib as mp
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf 
import matplotlib.pyplot as plt


# In[2]:


spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext


# In[3]:


tweet_raw = (spark.read.format("csv").options(header="true").load("20200312_Coronavirus_Tweets_Subset.CSV"))
display(tweet_raw)
tweet_raw.show(10, False)


# In[4]:


from pyspark.sql.functions import explode
from pyspark.sql import functions as F


# In[5]:


tweet_filter = tweet_raw.select("*", F.when(tweet_raw.lang == 'en', 'TRUE').alias('eng_true'))
tweet_filter.show()
tweet_filter = tweet_filter.filter("eng_true == 'TRUE'")
tweet_filter.show()


# In[8]:


tweet_filter.show(10)


# In[9]:


import textblob as tb


# In[10]:


pip install textblob


# In[11]:


import textblob as tb


# In[61]:


positive = 0
negative = 0
neutral = 0
polarity = 0


# In[62]:


def percent_calc(a, b):
    return 100 * float(a) / float(b)


# In[63]:


tweets_col = tweet_filter.select("text")
tweets = tweets_col.rdd.map(lambda row : row[0]).collect()
n_tweets = len(tweets_col.rdd.map(lambda row : row[0]).collect())


# In[74]:


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


# In[75]:


positive = percent_calc(positive, n_tweets)
negative = percent_calc(negative, n_tweets)
neutral = percent_calc(neutral, n_tweets)


# In[76]:


positive = format(positive, '.2f')
negative = format(negative, '.2f')
neutral = format(neutral, '.2f')


# In[22]:


print('----------------------------------------------------------------------------')
if polarity > 0:
    print('Positive')
elif polarity < 0:
    print('Negative')
elif polarity == 0:
    print('Neutral')


# In[23]:


labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]','Negative [' + str(negative) + '%]']
sizes = [positive, neutral, negative]
colors = ['green', 'yellow', 'red']
patches, texts = plt.pie(sizes, colors = colors, startangle = 90)
plt.legend(patches, labels, loc = "best")
plt.title('How people are reacting on ' + 'Covid-19' + ' by analyzing ' + str(n_tweets) + ' Tweets.')
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[24]:


tweet_raw_april = (spark.read.format("csv").options(header="true").load("2020-04-01 Coronavirus Tweets.CSV"))
display(tweet_raw_april)
tweet_raw_april.show(10, False)


# In[30]:


tweet_filter_april = tweet_raw_april.select("*", F.when(tweet_raw_april.lang == 'en', 'TRUE').alias('eng_true'))
tweet_filter_april.show()
tweet_filter_april = tweet_filter_april.filter("eng_true == 'TRUE'")
tweet_filter_april.show()


# In[28]:


from pyspark.sql.functions import explode
from pyspark.sql import functions as F


# In[31]:


tweets_col_april = tweet_filter_april.select("text")
tweets_april = tweets_col_april.rdd.map(lambda row : row[0]).collect()
n_tweets_april = len(tweets_col_april.rdd.map(lambda row : row[0]).collect())


# In[46]:


for tweetapril in tweets_april:
    print(tweetapril)
    myAnalysisApril = tb.TextBlob(tweetapril)
    polarity += myAnalysisApril.sentiment.polarity
    if myAnalysisApril.sentiment.polarity == 0:
        neutral += 1
    elif myAnalysisApril.sentiment.polarity > 0.00:
        positive += 1
    elif myAnalysisApril.sentiment.polarity < 0.00:
        negative += 1


# In[47]:


positive_april = percent_calc(positive, n_tweets_april)
negative_april = percent_calc(negative, n_tweets_april)
neutral_april = percent_calc(neutral, n_tweets_april)


# In[52]:


print(neutral_april)


# In[53]:


positive_april = format(positive_april, '.2f')
negative_april = format(negative_april, '.2f')
neutral_april = format(neutral_april, '.2f')


# In[56]:


labels = ['Positive [' + str(positive_april) + '%]', 'Neutral [' + str(neutral_april) + '%]','Negative [' + str(negative_april) + '%]']
sizes = [positive_april, neutral_april, negative_april]
colors = ['green', 'yellow', 'red']
patches, texts = plt.pie(sizes, colors = colors, startangle = 90)
plt.legend(patches, labels, loc = "best")
plt.title('How people are reacting on ' + 'Covid-19' + ' by analyzing ' + ' Tweets in April.')
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[85]:


print(positive) 
print(positive_april)
print(negative) 
print(negative_april)
print(neutral) 
print(neutral_april)


# In[93]:


compare={'Type':['Positive','Negative','Neutral'],
         'March 12 Percentage': [positive, negative, neutral],
         'April 1 Percentage':[positive_april, negative_april, neutral_april],
         'Percent Change':[float(positive_april)-float(positive), float(negative_april)-float(negative), float(neutral_april)-float(neutral)]
        }
compare_df=pd.DataFrame(compare)
compare_df


# In[ ]:


#Word Cloud Analysis


# In[95]:


pip install wordcloud 


# In[135]:


from wordcloud import WordCloud, STOPWORDS


# In[97]:


import pillow


# In[98]:


pip install pillow


# In[99]:


from PIL import Image


# In[103]:


marchdf=pd.DataFrame(tweets)


# In[143]:


marchdf.head()
text=str(marchdf)
print(text)


# In[150]:


stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords).generate(text)


# In[152]:


plt.imshow(wordcloud)
plt.figure(figsize = (10, 10), facecolor = None) 
plt.axis("off")


# In[153]:


aprildf=pd.DataFrame(tweets_april)


# In[157]:


aprildf.head()
text2=str(aprildf)
print(text2)


# In[155]:


stopwords=set(STOPWORDS)
wordcloud2=WordCloud(stopwords=stopwords).generate(text2)


# In[158]:


plt.imshow(wordcloud2)
plt.figure(figsize = (10, 10), facecolor = None) 
plt.axis("off")


# In[ ]:
#ALTERNATIVE WORDCLOUD CREATION USING PD DATAFRAME. STILL USING SUBSETTED CSV AND APRIL CSV
#Somewhat cleaner code

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
pip install wordcloud
from wordcloud import WordCloud, STOPWORDS


# In[10]:


marchdf=pd.read_csv("marchdata.CSV")


# In[11]:


marchdf=marchdf[marchdf.lang=="en"]


# In[12]:


marchdf=marchdf[["created_at",'text','screen_name','retweet_count','followers_count','friends_count']]


# In[13]:


marchdf.head()


# In[14]:


text=str(marchdf[["text"]])


# In[15]:


stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords).generate(text)


# In[16]:


plt.imshow(wordcloud)
plt.figure(figsize = (10, 10), facecolor = None)


# In[22]:


aprildf=pd.read_csv("aprildata.CSV")
aprildf=aprildf[aprildf.lang=="en"]
aprildf=aprildf[["created_at",'text','screen_name','retweet_count','followers_count','friends_count']]


# In[23]:


text2=str(aprildf[["text"]])


# In[26]:


stopwords=["text"]+list(STOPWORDS)
wordcloud2=WordCloud(stopwords=stopwords).generate(text2)


# In[27]:


plt.imshow(wordcloud2)
plt.figure(figsize = (10, 10), facecolor = None)




