import numpy as np
import pandas as pd
import utils

# For LSTM model
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback

# For hyperopt (parameter optimization)
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperopt.pyll.base import scope  # quniform returns float, some parameters require int; use this to force int

# mlflow
import mlflow
import mlflow.keras
import mlflow.tensorflow


def run():
    def f_nn(params):
        # Generate data with given window
        saltlake_week = pd.read_csv('C:/Users/aawang/CovidTraffic/Data/saltlake_week.csv')
        data = saltlake_week[['Cases', 'VMT (Veh-Miles)', 'News Sentiment', 'Unemployment Rate', 'PRCP', 'SNWD',
                              'Percent_Fully_Vaccinated_5&Older', 'TAVG',
                              'Stay at Home', 'Mask', 'School Opening', 'Health Emergency', 'Holiday']]
        X_train, y_train, X_valid, y_valid, X_test, y_test = utils.format_data(data=data, weeks=params['weeks'])

        with mlflow.start_run():
            mlflow.keras.autolog()
            # Keras LSTM model
            model = Sequential()

            if params['layers'] == 1:
                model.add(LSTM(units=params['units'], input_shape=(X_train.shape[1], X_train.shape[2]),
                               activation=params['activation']))
                model.add(Dropout(rate=params['dropout']))
            else:
                # First layer specifies input_shape and returns sequences
                model.add(
                    LSTM(units=params['units'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),
                         activation=params['activation']))
                model.add(Dropout(rate=params['dropout']))
                # Middle layers return sequences
                for i in range(params['layers'] - 2):
                    model.add(LSTM(units=params['units'], return_sequences=True, activation=params['activation']))
                    model.add(Dropout(rate=params['dropout']))
                # Last layer doesn't return anything
                model.add(LSTM(units=params['units'], activation=params['activation']))
                model.add(Dropout(rate=params['dropout']))

            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

            result = model.fit(X_train, y_train, verbose=0, validation_data=(X_valid, y_valid),
                               batch_size=4,
                               epochs=200,
                               callbacks=[es, TqdmCallback(verbose=1)]
                               )

            # get the lowest validation loss of the training epochs
            validation_loss = np.amin(result.history['val_loss'])
            print('Best validation loss of epoch:', validation_loss)
            mlflow.end_run()

        return {'loss': validation_loss, 'status': STATUS_OK, 'model': model, 'params': params}

    # hyperparameters to search over with hyperopt
    space = {'dropout': hp.uniform('dropout', 0.01, 0.5),
             'units': scope.int(hp.quniform('units', 10, 100, 5)),
             'layers': scope.int(hp.quniform('layers', 1, 6, 1)),
             'weeks': scope.int(hp.quniform('weeks', 1, 10, 1)),
             'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh'])
             }

    trials = Trials()
    best = fmin(f_nn, space, algo=tpe.suggest, max_evals=10, trials=trials)

    # get best model
    best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
    best_params = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']

    print(best_params)
    print(best_model.summary())
    best_model.save('Code/LSTM/Model/LSTM')


if __name__ == "__main__":
    run()

#  python Code/LSTM/search_hyperopt.py
