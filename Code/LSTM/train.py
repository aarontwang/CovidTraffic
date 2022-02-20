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

import mlflow
import mlflow.tensorflow

mlflow.tensorflow.autolog(log_models=True)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", required=False, help="path to output loss plot")
args = vars(ap.parse_args())

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

saltlake_week = pd.read_csv('C:/Users/aawang/CovidTraffic/Data/saltlake_week.csv')

X_total = saltlake_week[['Cases', 'VMT (Veh-Miles)', 'News Sentiment', 'Unemployment Rate', 'PRCP', 'SNWD',
                         'Percent_Fully_Vaccinated_5&Older', 'TAVG',
                         'Stay at Home', 'Mask', 'School Opening', 'Health Emergency', 'Holiday']].values[:156, :]

data = X_total[(60-config.N_WEEKS):, :]
scaler, values = utils.scale(data)
values = utils.series_to_supervised(values, n_in=config.N_WEEKS, n_out=1, dropnan=True).values

y_scaler, y = utils.scale(data[:, 1].reshape((len(data), 1)))

train = values[:88, :]
val = values[88:92]
test = values[92:, :]
print(train.shape)
print(val.shape)
print(test.shape)

y = values[:, -config.N_FEATURES:]

X_train = train[:, :config.N_OBS]
y_train = train[:, -config.N_FEATURES:][:, 1]
X_val = val[:, :config.N_OBS]
y_val = val[:, -config.N_FEATURES:][:, 1]
X_test = test[:, :config.N_OBS]
y_test = test[:, -config.N_FEATURES:][:, 1]

X_train = X_train.reshape((X_train.shape[0], config.N_WEEKS, config.N_FEATURES))
X_val = X_val.reshape((X_val.shape[0], config.N_WEEKS, config.N_FEATURES))
X_test = X_test.reshape((X_test.shape[0], config.N_WEEKS, config.N_FEATURES))

es = EarlyStopping(
    monitor="val_loss",
    patience=config.EARLY_STOPPING_PATIENCE,
    restore_best_weights=True)

model = keras.Sequential()
model.add(keras.layers.LSTM(40, input_shape=config.INPUT_SHAPE, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1))
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.01),
              metrics=['mse'])
model.build()

print("[INFO] training the model...")
with mlflow.start_run() as run:
    history = model.fit(
        x=X_train, y=y_train,
        validation_data=(X_val, y_val),
        batch_size=config.BS,
        callbacks=[es],
        epochs=config.EPOCHS
    )


print("[INFO] evaluating network...")
predictions = model.predict(x=X_test)
print("Loss and MSE: {}".format(model.evaluate(X_test, y_test)))

utils.save_plot(history, "output/loss_plot.png")
model.save('Model/LSTM')

y_test_inv = utils.invert_scale(y_scaler, y_test.reshape((len(y_test), 1)))
predictions_inv = y_scaler.inverse_transform(predictions)

X = values[:, :config.N_OBS]
y = values[:, -config.N_FEATURES:][:, 1]

m, n = X.shape

X = X.reshape((X.shape[0], config.N_WEEKS, config.N_FEATURES))
total_pred = model.predict(x=X)

y_inv = utils.invert_scale(y_scaler, y.reshape((len(y), 1)))
total_pred_inv = y_scaler.inverse_transform(total_pred)

mse_test = mean_squared_error(y_test_inv, predictions_inv)
rmse_test = mean_squared_error(y_test_inv, predictions_inv, squared=False)
mape_test = mean_absolute_percentage_error(y_test_inv, predictions_inv)

print("Mean Squared Error: {}".format(mse_test))
print("Root Mean Squared Error: {}".format(rmse_test))
print("Mean Absolute Percentage Error: {}".format(mape_test))

utils.plot_predicted(y_test_inv, predictions_inv, "output/vmt_test_predicted.png")
utils.plot_predicted(y_inv, total_pred_inv, "output/vmt_total_predicted.png")

# python train.py --plot Code/LSTM/output/loss_plot.png
# mlflow run -e train.py --experiment-id 1 Code/LSTM --no-conda
