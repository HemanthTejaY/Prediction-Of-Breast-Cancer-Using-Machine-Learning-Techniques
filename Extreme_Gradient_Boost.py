# coding: utf-8
# In[2]:
import pyspark
import pandas as pdfrom pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.sql.functions import *

# In[3]:
sc=pyspark.SparkContext.getOrCreate()
sqlContext=SQLContext(sc)
#Reading Data
df=pd.read_csv('breastwisconsin.csv')
df

# In[4]:
# rd = radius , tx = texture , pm = perimeter , ar = area , sm = smoothness , cn =
compactness, cc = concavity , cp = concave points , sym = symmetry , fd = fractal
dimension
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
StructField('fd',StringType(),True)])

# In[7]:
dfs=sqlContext.createDataFrame(df,schema)
dfs

# In[8]:
from pyspark.sql.functions import *
dfs.take(2)
dfs.printSchema()
# In[9]:
dfs.first()
dfs = dfs.select("rd","tx","pm","ar","sm","cn","cc","cp","sym","fd")dfs.describe().show()

# In[10]:
from pyspark.ml.feature import StringIndexer,VectorAssembler,IndexToString
labelindexer = StringIndexer(inputCol = "fd", outputCol = "label").fit(dfs)
featureassembler = VectorAssembler(inputCols =["rd","tx","pm","ar","sm","cn","cc","cp","sym"], outputCol = "features")
featureassembler

# In[11]:
train_data, test_data=dfs.randomSplit([.8,.2],seed=1234)

# In[12]:
#from pyspark.ml.regression import LinearRegression
#from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
train_data.printSchema()
train_data.show()
#nb=NaiveBayes(smoothing=1.0,modelType="multinomial")
#lr=LinearRegression(labelCol="label",featuresCol="features",maxIter=10,regParam=0.3,el
asticNetParam=0.8)

# In[14]:
#mlp = MultilayerPerceptronClassifier(maxIter=50,featuresCol = "features",labelCol
="label", layers=[10,5,2], blockSize=128, seed=None)
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
#stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
gbt = GBTClassifier(maxIter = 10 ,labelCol="label",featuresCol = "features")

# In[15]:
pipeline = Pipeline(stages = [labelindexer,featureassembler,gbt])

# In[21]:
model=pipeline.fit(train_data)

# In[22]:
predictions = model.transform(test_data)
predictions

# In[23]:from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol = "label",predictionCol ="prediction",metricName = "accuracy")

# In[24]:
accuracy1 = evaluator.evaluate(predictions)
print(accuracy1)
sc.stop()
