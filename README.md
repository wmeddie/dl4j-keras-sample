# Example os using DL4J's spark training from Python

## Step 1:  Build a DL4J uberjar.

With maven installed in the computer just run 

    mvn clean package

## Step 2: Package python dependencies for Spark.

Just run the following:

    python setup.py bdist_spark

## Step 3: Run the job on a spark cluster.*

Run spark-submit and specify the built dl4j uberjar and python zip files.

    spark-submit --driver-memory 8g --master yarn --jars target/dl4j-uber-1.0.0-beta4.jar --py-files spark_dist/keras_dl4j_sample-0.1-deps.zip,spark_dist/keras_dl4j_sample-0.1.zip keras_dl4j_sample/driver.py train.spark_script

*: Training on a cluster not yet working as some beta4 APIs have changed.  Coming soon.