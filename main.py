import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping

from prediction import predict_func
from plotting_graphs import plot_graphs_func
from future_predict import future_pred


def main():
    print(f'Hi, PyCharm')

    # Input csv file
    directory = input("Enter data file to train: ")
    #/home/cranberry/Desktop/Stock prediction model/Axis bank test/AXISBANK.csv
    df = pd.read_csv(directory)
    directory2 = input("Enter data file to manual test: ")
    #/home/cranberry/Downloads/AXISBANK_latest.csv
    df1 = pd.read_csv(directory2)

    df['Date'] = pd.to_datetime(df['Date'])

    # Setting target variable and features
    op_var = input("Select target variable: ")
    output_var = pd.DataFrame(df[op_var])
    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    features.remove(op_var)

    # Scaling
    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(df[features])
    feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=df.index)

    tvar_transform = scaler.fit_transform(output_var)
    tvar_transform = pd.DataFrame(tvar_transform)

    # Splitting to Training set and Test set
    training_size = int(len(feature_transform) * 0.65)
    test_size = len(feature_transform) - training_size
    train_data = tvar_transform[training_size:len(feature_transform)]

    Xtrain_data, Xtest_data = feature_transform[0:training_size], feature_transform[training_size:len(feature_transform)]
    ytrain_data, ytest_data = tvar_transform[0:training_size], tvar_transform[training_size:len(tvar_transform)]

    # Process the data for LSTM
    time_step = 100
    X_train, y_train = create_dataset(Xtrain_data, ytrain_data, time_step)
    X_test, y_test = create_dataset(Xtest_data, ytest_data, time_step)

    # Building the LSTM Model
    number_days = int(input("How many days to predict?: "))

    lstm = Sequential()
    lstm.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=False))
    #lstm.add(LSTM(16, return_sequences=False))
    #lstm.add(LSTM(100))
    lstm.add(Dense(number_days))
    lstm.compile(loss='mean_squared_error', optimizer='adam')

    # Model Training
    print("Model training starts...")

    choice = input("Do you want to train the model? [y/n]:")

    if choice == 'y':
        earlyStop = EarlyStopping(monitor="loss", verbose=1, mode='min', patience=10)
        history = lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=10, verbose=1, shuffle=False)

    # LSTM Prediction
    y_pred_test = lstm.predict(X_test)
    y_pred_train = lstm.predict(X_train)
    y_pred_test = scaler.inverse_transform(y_pred_test)
    y_pred_train = scaler.inverse_transform(y_pred_train)

    # Plotting
    #plot_graphs_func(target, y_pred_scaled, [])

    # future prediction

    looper = 'y'

    while looper == 'y':

        y_pred_future = lstm.predict(X_test[-1:])
        y_pred_future = scaler.inverse_transform(y_pred_future)

        # Plotting
        plt.plot(df['Close'], label='True value')
        plt.plot(range(100, 100 + len(y_pred_train)), y_pred_train[:, 0], label='Training set')
        plt.plot(range(903, 903 + len(y_pred_test)), y_pred_test[:, 0], label='Test set')
        plt.plot(range(1233, 1233 + len(y_pred_future[0])), y_pred_future[0], label='Future Prediction')
        plt.plot(range(1233, 1233 + len(df1['Close'])), df1['Close'], label='Untrained values')
        plt.title('Prediction by LSTM')
        plt.xlabel('Time Scale')
        plt.ylabel('Scaled Rupees')
        plt.legend()
        plt.show()

        #plot_graphs_func(target, y_pred_scaled, y_pred_future)

        looper = input("Do you want to predict again? [y/n]: ")

    print("End of program")


def create_dataset(train_dataset, test_dataset, time_step=1):
    dataX = []
    dataY = []
    tds = np.array(test_dataset)
    for i in range(len(train_dataset)-time_step-1):
        a = train_dataset[i:(i+time_step)]
        b = tds[i][0]
        dataX.append(a)
        dataY.append(b)
    return np.array(dataX), np.array(dataY)


if __name__ == '__main__':
    main()
