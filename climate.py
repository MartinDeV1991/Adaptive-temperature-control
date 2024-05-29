import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


import tensorflow as tf
import keras


# Prepare time series data
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i : (i + seq_length)].drop("date", axis=1).values
        y = data.iloc[i + seq_length]["temperature"]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def data_prep():
    # Simulate sample data
    np.random.seed(42)
    dates = pd.date_range("20240101", periods=5000, freq="H")

    # temperature = 20 + 2 * np.sin(2 * np.pi * (dates.hour + dates.dayofyear * 24) / 24)
    # noise = np.random.normal(0, 0.5, size=temperature.shape)
    # temperature += noise
    # temperature = np.array(temperature)
    # days_of_week = dates.dayofweek
    # temperature[days_of_week == 3] -= 2

    temperature = np.zeros(len(dates))
    for i, date in enumerate(dates):
        if date.dayofweek == 2:  # Wednesday
            if 9 <= date.hour < 21:
                temperature[i] = 20
            else:
                temperature[i] = 18
        elif date.dayofweek == 6:
            if 13 <= date.hour < 23:
                temperature[i] = 21
            elif 11 <= date.hour < 13:
                temperature[i] = 19.5
            elif 23 <= date.hour < 24:
                temperature[i] = 19.5
            else:
                temperature[i] = 18
        else:
            if 9 <= date.hour < 21:
                temperature[i] = 21
            else:
                temperature[i] = 18

    # noise = np.random.normal(0, 0.1, size=temperature.shape)
    # temperature += noise

    data = pd.DataFrame({"date": dates, "temperature": temperature})

    # Normalize the data
    scaler = MinMaxScaler()
    data["temperature"] = scaler.fit_transform(data[["temperature"]])

    seq_length = 24
    X, y = create_sequences(data, seq_length)

    # Split data into train and test sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test, seq_length, scaler, data, split


def model_building(seq_length, X_train, y_train):
    # Build the LSTM model
    # model = keras.Sequential()
    # model.add(keras.layers.LSTM(50, activation="relu", input_shape=(seq_length, 1)))
    # model.add(keras.layers.Dense(1))  # Output for temperature

    model = keras.Sequential(
        [
            keras.layers.LSTM(
                64,
                input_shape=(X_train.shape[1], X_train.shape[2]),
                return_sequences=True,
            ),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")

    # history = model.fit(X_train, y_train, epochs=20, validation_split=0.2)
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    return model, history


def evaluation(model, X_test, y_test, scaler, data, split):
    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")

    # Make predictions
    y_pred = model.predict(X_test)

    # Rescale the predictions and actual values to original scale
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    test_dates = data["date"].iloc[split + seq_length :].reset_index(drop=True)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates[0:200], y_test_rescaled[0:200], label="Actual Temperature")
    plt.plot(test_dates[0:200], y_pred_rescaled[0:200], label="Predicted Temperature")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.legend()
    plt.xticks(rotation=45)
    plt.title("Actual vs Predicted Temperature")
    plt.show()


X_train, X_test, y_train, y_test, seq_length, scaler, data, split = data_prep()
model, history = model_building(seq_length, X_train, y_train)
evaluation(model, X_test, y_test, scaler, data, split)
