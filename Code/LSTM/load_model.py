import argparse
import config
import math
import os
import random
import utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing

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

X_total = saltlake_week[['Cases', 'VMT (Veh-Miles)', 'Avg News Sentiment', 'Unemployment Rate', 'PRCP', 'SNOW',
                         'Percent_Fully_Vaccinated_5&Older', 'TAVG',
                         'Stay at Home', 'Mask', 'School Opening', 'Health Emergency']].values

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(X_total[58:148, :])
X_total_scaled = scaler.transform(X_total[58:148, :])
data = utils.series_to_supervised(X_total_scaled, n_in=config.N_WEEKS, n_out=1).values

print(data.shape)

X = data[:, :config.N_OBS]
y = data[:, -config.N_FEATURES:]

X_train = X[:84, :]
y_train = y[:84, :]
X_test = X[84:, :]
y_test = y[84:, :]

X_train = X_train.reshape((X_train.shape[0], config.N_WEEKS, config.N_FEATURES))
X_test = X_test.reshape((X_test.shape[0], config.N_WEEKS, config.N_FEATURES))

es = EarlyStopping(
    monitor="val_loss",
    patience=config.EARLY_STOPPING_PATIENCE,
    restore_best_weights=True)

model = keras.models.load_model(args['model'])

model.reset_states()
print("[INFO] model summary")
print(model.summary())

print("[INFO] evaluating network...")
print("Loss and MSE: {}".format(model.evaluate(X_test, y_test)))

ythat = model.predict(X_train, batch_size=4)
model.reset_states()
yhat = model.predict(X_test, batch_size=4)
model.reset_states()

# Rescale values
train_rescref = scaler.inverse_transform(ythat)
test_rescref = scaler.inverse_transform(yhat)

# all test result visualization

test_pred = test_rescref[:, 1]
test_true = X_total[144:148, 1]

train_pred = train_rescref[:, 1]
train_true = X_total[60:144, 1]

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

# python load_model.py --model Model/LSTM --plot output/training_plot.png
