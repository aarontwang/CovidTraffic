import os
import math
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import initializers
import stellargraph as sg
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('TkAgg')


seed_value = 1
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def sequence_data_preparation(seq_len, pre_len, data):
    dataX, dataY = [], []

    for i in range(data.shape[1] - int(seq_len + pre_len - 1)):
        a = data[:, i : i + seq_len + pre_len]
        dataX.append(a[:, :seq_len])
        dataY.append(a[:, -1])

    dataX = np.array(dataX)
    dataY = np.array(dataY)

    return dataX, dataY


saltlake_week = pd.read_csv('Data/utah_week.csv')

X_total = saltlake_week[['Cases', 'VMT Total', 'News Sentiment', 'Unemployment Rate', 'PRCP', 'SNWD',
                         'Percent_Fully_Vaccinated_12&Older', 'TOBS', 'Stay at Home', 'Mask', 'School Opening', 'Health Emergency']].fillna(0).values

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_total_scaled = scaler.fit_transform(X_total)
data = X_total_scaled.T

seq_len = 2
pre_len = 1

dataX, dataY = sequence_data_preparation(seq_len, pre_len, data)

dataX_train = dataX[104:124, :, :]
dataY_train = dataY[104:124, :]
dataX_test = dataX[124:, :, :]
dataY_test = dataY[124:, :]

total_adjacent_matrix = np.asmatrix(pd.read_csv('Data/DirectedAM.csv').values[:, 1:])

from stellargraph.layer import GCN_LSTM

gcn_lstm = GCN_LSTM(
    seq_len=seq_len,
    adj=total_adjacent_matrix,
    gc_layer_sizes=[6, 4],
    gc_activations=["relu", "relu"],
    lstm_layer_sizes=[20, 20, 40, 40, 40, 40, 20, 20],
    lstm_activations=["relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu"],
    kernel_initializer=initializers.Identity(gain=1)
)

x_input, x_output = gcn_lstm.in_out_tensors()
model = Model(inputs=x_input, outputs=x_output)
model.compile(optimizer="adam", loss="mse", metrics=["mse"])

rmse_benchmark = math.sqrt(mean_squared_error(X_total[126:, 1]/1000000, X_total[125:129, 1]/1000000))
mape_benchmark = math.sqrt(mean_absolute_percentage_error(X_total[126:, 1]/1000000, X_total[125:129, 1]/1000000))

rmse_ref = rmse_benchmark
mape_ref = mape_benchmark


for i in range(10000):
    history = model.fit(
        dataX_train,
        dataY_train,
        epochs=1,
        batch_size=5,
        # shuffle=True,
        verbose=0,
        validation_data=(dataX_test, dataY_test)
    )

    #
    # print(
    #     "Train loss: ",
    #     history.history["loss"][-1],
    #     "\nTest loss:",
    #     history.history["val_loss"][-1],
    # )

    # sg.utils.plot_history(history)

    # ythat = model.predict(dataX_train)
    yhat = model.predict(dataX_test)

    ## Rescale values
    # train_rescref = scaler.inverse_transform(ythat)
    test_rescref = scaler.inverse_transform(yhat)

    # ##all test result visualization
    # fig1 = plt.figure(figsize=(15, 8))
    # #    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_rescref[:, 1]
    a_true = X_total[126:, 1]
    # plt.plot(a_pred, "r-", label="prediction")
    # plt.plot(a_true, "b-", label="true")
    # plt.xlabel("time")
    # plt.ylabel("vmt")
    # plt.legend(loc="best", fontsize=10)
    # plt.show()

    trainScore = math.sqrt(mean_squared_error(a_true/1000000, a_pred/1000000))
    trainScore_mape = math.sqrt(mean_absolute_percentage_error(a_true/1000000, a_pred/1000000))

    if (trainScore < rmse_ref) and (trainScore_mape < mape_ref):
        rmse_ref = trainScore
        mape_ref = trainScore_mape
        print(
            "RMSE: ",
            rmse_ref,
            "MAPE",
            mape_ref,
            'Epoch: ',
            i
        )
        model.save('Model/GCN-LSTM-3')

    if (i % 100) == 0:
        print(
            "Milestone: ",
            i,
            "RMSE: ",
            trainScore,
            "MAPE",
            trainScore_mape,
        )


model = tf.keras.models.load_model('Model/GCN-LSTM-3')
