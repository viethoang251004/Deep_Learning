import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('AirPassengers.csv')

# Assume that 'Month' is your time variable and '#Passengers' is your target variable
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Define the model
model = ARIMA(df, order=(5,1,0))

# Fit the model
model_fit = model.fit()

# Make prediction
prediction = model_fit.predict(start=len(df), end=len(df)+12)

# Plot the original data and the forecast
plt.plot(df)
plt.plot(prediction)
plt.show()
