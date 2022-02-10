import argparse
import config
import math
import os
import random
import utils
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping
from keras import Model, initializers
from sklearn import preprocessing

import stellargraph as sg
from stellargraph.layer import GCN_LSTM

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to model")
ap.add_argument("-p", "--plot", required=True, help="path to output loss plot")
args = vars(ap.parse_args())

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

saltlake_week = pd.read_csv('../Data/saltlake_week.csv')

saltlake_week.fillna(0, inplace=True)

X_total = saltlake_week[['Cases', 'VMT (Veh-Miles)', 'News Sentiment', 'Unemployment Rate', 'PRCP', 'SNWD',
                         'Percent_Fully_Vaccinated_5&Older', 'TAVG',
                         'Stay at Home', 'Mask', 'School Opening', 'Health Emergency']].values

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(X_total[59:148, :])
X_total_scaled = scaler.transform(X_total[59:148, :])
data = X_total_scaled.T
print(data.shape)
X, y = utils.sequence_data_preparation(config.SEQ_LEN, config.PRE_LEN, data)

X_train = X[:84, :, :]
y_train = y[:84,:]
X_test = X[84:, :, :]
y_test = y[84:, :]

es = EarlyStopping(
    monitor="val_loss",
    patience=config.EARLY_STOPPING_PATIENCE,
    restore_best_weights=True)

model = keras.models.load_model(args['model'])

x_input, x_output = model.in_out_tensors()
model = Model(inputs=x_input, outputs=x_output)
lr = 0.001
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse",
              metrics=["mse", keras.metrics.RootMeanSquaredError()])

model.reset_states()
print("[INFO] model summary")
print(model.summary())

print("[INFO] evaluating network...")
print("Loss and MSE: {}".format(model.evaluate(X_test, y_test)))

ythat = model.predict(X_train)
model.reset_states()
yhat = model.predict(X_test)
model.reset_states()

# Rescale values
train_rescref = scaler.inverse_transform(ythat)
test_rescref = scaler.inverse_transform(yhat)

# all test result visualization

test_pred = test_rescref[:, 1]
test_true = X_total[144:148, 1]

train_pred = train_rescref[:, 1]
train_true = X_total[59:144, 1]

trainScore = math.sqrt(mean_squared_error(test_true / 1000000, test_pred / 1000000))
trainScore_mape = mean_absolute_percentage_error(test_true / 1000000, test_pred / 1000000)

print("Root Mean Squared Error: {}".format(trainScore))
print("Mean Absolute Percentage Error: {}".format(trainScore_mape))

fig1 = plt.figure(figsize=(15, 8))
ax1 = fig1.add_subplot(1, 1, 1)
plt.plot(test_pred, "r-", label="prediction")
plt.plot(test_true, "b-", label="true")
plt.xlabel("time")
plt.ylabel("vmt")
plt.legend(loc="best", fontsize=10)
plt.show()

fig2 = plt.figure(figsize=(15, 8))
ax2 = fig2.add_subplot(1, 1, 1)
plt.plot(train_pred, "r-", label="prediction")
plt.plot(train_true, "b-", label="true")
plt.xlabel("time")
plt.ylabel("vmt")
plt.legend(loc="best", fontsize=10)
plt.show()

utils.plot_predicted(test_true, test_pred, "output/vmt_test_predicted.png")
utils.plot_predicted(train_true, train_pred, "output/vmt_train_predicted.png")

# python load_model.py --model Model/GCN-LSTM- --plot output/loss_plot.png
