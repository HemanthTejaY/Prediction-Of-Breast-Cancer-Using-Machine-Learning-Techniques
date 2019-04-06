# coding: utf-8# In[1]:
import pyspark
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.sql.functions import *

# In[2]:
sc=pyspark.SparkContext.getOrCreate()
sqlContext=SQLContext(sc)
#Reading Data
df=pd.read_csv('breastwisconsin.csv')


# In[3]:
schema= StructType([StructField('id',IntegerType(),True) ,
StructField('rd',IntegerType(),True) ,
StructField('tx',IntegerType(),True) ,
StructField('pm',IntegerType(),True) ,
StructField('ar',IntegerType(),True) ,
StructField('sm',IntegerType(),True) ,
StructField('cn',IntegerType(),True) ,
StructField('cc',IntegerType(),True) ,
StructField('cp',IntegerType(),True) ,
StructField('sym',IntegerType(),True) ,
StructField('fd',StringType(),True)])'

# In[4]:
dfs=sqlContext.createDataFrame(df,schema)

# In[5]:
from pyspark.sql.functions import *
dfs.take(2)
dfs.printSchema()

# In[6]:
dfs.first()
dfs = dfs.select("rd","tx","pm","ar","sm","cn","cc","cp","sym","fd")
dfs.describe().show()

# In[7]:
from pyspark.ml.feature import StringIndexer,VectorAssembler,IndexToString
labelindexer = StringIndexer(inputCol = "fd", outputCol = "label").fit(dfs)featureassembler = VectorAssembler(inputCols =
["rd","tx","pm","ar","sm","cn","cc","cp","sym"], outputCol = "features")
featureassembler

# In[8]:
train_data, test_data=dfs.randomSplit([.8,.2],seed=1234)

# In[10]:
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

# In[11]:
nb=NaiveBayes(labelCol="label",featuresCol =
"features",smoothing=1.0,modelType="multinomial")

# In[12]:
pipeline = Pipeline(stages = [labelindexer,featureassembler,nb])

# In[13]:
model=pipeline.fit(train_data)

# In[14]:
predictions = model.transform(test_data)

# In[15]:
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol = "label",predictionCol ="prediction",metricName = "accuracy")

# In[16]:
accuracy1 = evaluator.evaluate(predictions)
print(accuracy1)
sc.stop()
