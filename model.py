from tensorflow import keras

import config


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=hp.Int('lstm_1', min_value=10,
                                             max_value=100,
                                             step=10),
                                input_shape=config.INPUT_SHAPE))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(13))

    lr = hp.Choice('learning_rate',
                   values=[1e-1, 3e-1, 1e-2, 3e-2, 1e-3, 3e-3, 1e-4, 3e-4, 1e-5])

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr),
                  metrics=['mse'])
    return model
