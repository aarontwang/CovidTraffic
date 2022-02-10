import argparse
import config
import os
import random
import utils

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", required=True, help="path to output loss plot")
args = vars(ap.parse_args())

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

saltlake_week = pd.read_csv('../Data/saltlake_week.csv')

X_total = saltlake_week[['Cases', 'VMT (Veh-Miles)', 'News Sentiment', 'Unemployment Rate', 'PRCP', 'SNWD',
                         'Percent_Fully_Vaccinated_5&Older', 'TAVG',
                         'Stay at Home', 'Mask', 'School Opening', 'Health Emergency']].values[:156, :]

data = X_total[59:, :]
scaler, values = utils.scale(data)
values = utils.series_to_supervised(values, n_in=config.N_WEEKS, n_out=1, dropnan=True).values
X = values[:, :config.N_OBS]
y = values[:, -config.N_FEATURES:]

m, n = X.shape

train = values[:88, :]
test = values[88:, :]

X_train = train[:, :config.N_OBS]
y_train = train[:, -config.N_FEATURES:]

X_test = test[:, :config.N_OBS]
y_test = test[:, -config.N_FEATURES:]

X_train = X_train.reshape((X_train.shape[0], config.N_WEEKS, config.N_FEATURES))
X_test = X_test.reshape((X_test.shape[0], config.N_WEEKS, config.N_FEATURES))

es = EarlyStopping(
    monitor="val_loss",
    patience=config.EARLY_STOPPING_PATIENCE,
    restore_best_weights=True)

model = keras.Sequential()
model.add(keras.layers.LSTM(20, input_shape=config.INPUT_SHAPE, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(12))
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.01),
              metrics=['mse'])

print("[INFO] training the model...")
H = model.fit(
    x=X_train, y=y_train,
    validation_split=0.2,
    batch_size=config.BS,
    callbacks=[es],
    epochs=config.EPOCHS
)

print("[INFO] evaluating network...")
predictions = model.predict(x=X_test)
print("Loss and MSE: {}".format(model.evaluate(X_test, y_test)))

utils.save_plot(H, args["plot"])
model.save('Model/LSTM')

y_test_inv = utils.invert_scale(scaler, y_test)[:, 1]
predictions_inv = scaler.inverse_transform(predictions)[:, 1]

X = X.reshape((X.shape[0], config.N_WEEKS, config.N_FEATURES))
total_pred = model.predict(x=X)

y_inv = utils.invert_scale(scaler, y[:, -config.N_FEATURES:])[:, 1]
total_pred_inv = scaler.inverse_transform(total_pred)[:, 1]

mse_test = mean_squared_error(y_test_inv, predictions_inv)
rmse_test = mean_squared_error(y_test_inv, predictions_inv, squared=False)
mape_test = mean_absolute_percentage_error(y_test_inv, predictions_inv)

print("Mean Squared Error: {}".format(mse_test))
print("Root Mean Squared Error: {}".format(rmse_test))
print("Mean Absolute Percentage Error: {}".format(mape_test))

utils.plot_predicted(y_test_inv, predictions_inv, "output/vmt_test_predicted.png")
utils.plot_predicted(y_inv, total_pred_inv, "output/vmt_total_predicted.png")

# python train.py --plot output/loss_plot.png
