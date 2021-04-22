from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc

# load csv data as pandas dataframe
dftrain = pd.read_csv("data/titanic/train.csv")
dfeval = pd.read_csv("data/titanic/eval.csv")
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")

# define column data forms
CATEGORICAL_COLUMNS = ["sex", "n_siblings_spouses", "parch", "class", "deck"]
NUMERIC_COLUMNS = ["age", "fare"]

# create feature column
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() # only get the unique data entries
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)) # append

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# train model
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) # create tf.data.Dataset object with data
                                                                           # and its labels
        if shuffle:
            ds = ds.shuffle((1000)) # randomize data order
        ds = ds.batch(batch_size).repeat(num_epochs) # split dataset into batches of 32 and repeat process epochs times
        return ds
    return input_function
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# create the linear regression model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# train the model
linear_est.train(train_input_fn)

# compare trained model to the real result data
result = linear_est.evaluate(eval_input_fn)

# show accuracy of the model
print(result["accuracy"])

# show individual information, actual result and model prediction
print(dfeval.loc[5])  # person information
print(f"survived: {y_eval[5]}")
result = list(linear_est.predict(eval_input_fn))
print(result[5]["probabilities"][1])
