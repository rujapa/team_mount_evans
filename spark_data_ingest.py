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
