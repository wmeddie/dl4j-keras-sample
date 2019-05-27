import os
import sys
from pyspark import SparkContext, SparkConf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils

def process(sc, args):
    print('importing elephas classes')
    from elephas.dl4j import ParameterSharingModel
    from elephas.utils.rdd_utils import to_java_rdd
    MnistDataSetIterator = sc._jvm.org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
    ArrayList = sc._jvm.java.util.ArrayList

    print('doing some spark stuff')
    input_data = args if args else [1, 2, 3, 4, 5]
    distr_data = sc.parallelize(input_data)
    result = distr_data.collect()
 
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    #print('Converting numpy array to rdd.')
    #train_rdd = to_java_rdd(sc._jsc, x_train, y_train, 64)

    print('Creating DL4J MNIST RDD')
    iter_train = MnistDataSetIterator(64, True, 12345)
    #iter_test = MnistDataSetIterator(64, False, 12345)

    train_data_list = ArrayList()
    #test_data_list = []

    while iter_train.hasNext():
        train_data_list.add(iter_train.next())
    #while iter_test.hasNext():
    #    test_data_list.append(iter_test.next())

    train_rdd = sc._jsc.parallelize(train_data_list)
    #test_rdd = sc.parallelize(test_data_list)


    print('building model')
    model = Sequential()
    model.add(Dense(128, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['acc'])

    print('converting model')
    spark_model = ParameterSharingModel(sc._jsc, model, batch_size=64, num_workers=4)

    print('spark_model has java_spark_model of: %s' % repr(spark_model.java_spark_model))

    #print('Setting controller address to: 10.0.0.21')
    #spark_model.java_spark_model.getTrainingMaster().setControllerAddress("10.0.0.21")


    print('training model...')
    spark_model.fit_rdd(train_rdd, epochs=20)
    print('training model finished.')

    # XXX: This is probably not the same held-out test set.
    score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)

    print('Test Accuracy: %s', score[1])

    spark_model.master_network.save("test_model.h5")

def main(args):
    job_name = 'distributed-keras-train'
    app_name = '{0}-{1}'.format(job_name, 'batchId')

    print('initialize context')
    sparkConf = SparkConf()
    sparkConf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator")

    sc = SparkContext(appName=app_name, conf=sparkConf)
    sc.setLogLevel('WARN')

    Nd4j = sc._jvm.org.nd4j.linalg.factory.Nd4j
    print(Nd4j.eye(3).toString())

    print('starting process...')
    process(sc, args)
    print('process finished.')

    sc.stop()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
