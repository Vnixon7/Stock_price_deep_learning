import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import tensorflow.keras
import keras

#function to plot model data
def plot_data(actual_prices, predicted_prices, company):
    plt.plot(actual_prices, color="red", label=f" Actual Price")
    plt.plot(predicted_prices, color="green", label=" Predicted Price")
    plt.title(f"{company} share price")
    plt.xlabel('Time')
    plt.ylabel(f"{company} share price")
    plt.legend()
    plt.show()



#Getting our data for model
company = 'GME'

#stock = yf.Ticker(company)
#print(stock.history('MAX'))

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)
data = web.DataReader(company, 'yahoo', start, end)



scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

#how many days in the future you want to predict the stock price
prediction_days = 60

X_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    X_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])


X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



#building model

#loading saved model
model = keras.models.load_model(r"C:\Users\Vnixo\OneDrive\Desktop\Stock_market_NN_model\RNN_test3")

#creating new model for testing
model = Sequential()

model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #output prediction of next day price

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#'''

#training model
model.fit(X_train, y_train, epochs=300, batch_size=64)
#'''

#saving model
model.save(r"C:\Users\Vnixo\OneDrive\Desktop\Stock_market_NN_model\RNN_test5")

#Test model acc of exiting data

#load test data


test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)



#predictions : test data


x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])


x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)

predicted_prices = scaler.inverse_transform(predicted_prices)

#print(f"actual:{actual_prices}")
#print(f"predicted:{predicted_prices}")
actual_df = pd.DataFrame(actual_prices)
predicted_df = pd.DataFrame(predicted_prices)
results = pd.concat([actual_df, predicted_df], axis=1)
results.columns = ['Actual_Price', 'Predicted_price']
#results.to_csv(fr'C:\Users\Vnixo\OneDrive\Desktop\{company}_stock_predictions.csv')
#print(result)

#plot

plot_data(actual_prices, predicted_prices, company)
#predicting next day
# + 1 to get prediction for one day in future
#'''
predict_results = []
days = 60
for i in range(1, days):

    real_data = [model_inputs[len(model_inputs) + i - prediction_days:len(model_inputs+1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    #print(scaler.inverse_transform(real_data[-1]))
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    predict_results.append(prediction[0])
    print(f"Prediction:{company}->{prediction[0]}")

#sending price predictions to csv
p_results = pd.DataFrame(predict_results)
p_results.columns = ['prediction']
p_results.to_csv(fr"C:\Users\Vnixo\OneDrive\Desktop\{company}_{days}_predictions.csv")
#'''



