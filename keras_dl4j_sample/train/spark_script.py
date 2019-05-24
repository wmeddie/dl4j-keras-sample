import os
import sys
from pyspark import SparkContext
import elephas
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils

from elephas.spark_model import SparkModel
from elephas.dl4j import ParameterSharingModel
from elephas.utils.rdd_utils import to_java_rdd

def process(sc, args):
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
    
    train_rdd = to_java_rdd(sc, x_train, y_train, 64)

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

    spark_model = ParameterSharingModel(sc._jsc, model, batch_size=64, num_workers=4)

    spark_model.fit_rdd(train_rdd, epochs=20)

    score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)

    print('Test Accuracy: %s', score[1])

    spark_model.master_network.save("test_model.h5")

def main(args):
    job_name = 'distributed-keras-train'
    app_name = '{0}-{1}'.format(job_name, 'batchId')

    print('initialize context')
    sc = SparkContext(appName=app_name)
    sc.setLogLevel('WARN')
    process(sc, args)
    sc.stop()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
