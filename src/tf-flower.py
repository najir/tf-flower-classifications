from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

CSV_COLUMNS = ['SepalLength', 'SepalWidth', 'PetalWidth', 'PetalLength', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

#Get files or filepatchs for our datasets
TrainPath = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
)
evalPath = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
)

#Import our datasets and wrap them in a pandas datafram
trainDf = pd.read_csv(trainPath, names=CSV_COLUMNS, header=0)
evalDf = pd.read_csv(evalPath, names=CSV_COLUMNS, header=0)

#Remove the variable we will be searching for to seperate it into our y-value
trainY = trainDf.pop('species')
evalY = evalDf.pop('species')

#Set up our feature columns
featureCol = []
for index in trainDf.keys():
    featureCol.append(tf.feature_column.numeric_column(key=index))





#Define our input function, batch size 256 in this case, shuffling the dataset 1000 times
def InputFunction(features, labels, training=true, batches=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batches)