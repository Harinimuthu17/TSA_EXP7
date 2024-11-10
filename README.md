## Name: M.HARINI
## Reg No: 212222240035
## Date: 

# Ex.No: 07  AUTO REGRESSIVE MODEL


## AIM:
To Implementat an Auto Regressive Model using Python on Tesla stock prediction.

## ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
   
## PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/content/Microsoft_Stock.csv')
print(data)

# Convert 'Date' to datetime format and set it as the index
data['date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Check for stationarity using the Augmented Dickey-Fuller (ADF) test on 'Volume'
result = adfuller(data['Volume'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split data into training and testing sets (80% training, 20% testing)
train_data = data.iloc[:int(0.8 * len(data))]
test_data = data.iloc[int(0.8 * len(data)):]

# Define the lag order for the AutoRegressive model (adjust lag based on ACF/PACF plots)
lag_order = 13
model = AutoReg(train_data['Volume'], lags=lag_order)
model_fit = model.fit()

# Plot Autocorrelation Function (ACF) for 'Volume'
plt.figure(figsize=(10, 6))
plot_acf(data['Volume'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Volume')
plt.show()

# Plot Partial Autocorrelation Function (PACF) for 'volume'
plt.figure(figsize=(10, 6))
plot_pacf(data['volume'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - volume')
plt.show()

# Make predictions on the test set
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Calculate Mean Squared Error (MSE) for the test set predictions
mse = mean_squared_error(test_data['Volume'], predictions)
print('Mean Squared Error (MSE):', mse)
# Plot Test Data vs Predictions for 'Volume'
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['volume'], label='Test Data - volume', color='purple', linewidth=2)
plt.plot(test_data.index, predictions, label='Predictions - volume', color='black', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('AR Model Predictions vs Test Data (Volume)')
plt.legend()
plt.grid(True)
plt.show()
```

## OUTPUT:

GIVEN DATA

![given data](https://github.com/user-attachments/assets/399ea8c0-bea3-4f9f-8708-02e88d058d38)

AUGMENTED DICKEY-FULLER TEST:

![Capture](https://github.com/user-attachments/assets/73a3bfad-a127-4acf-9624-fb8dc359603d)


PACF - ACF

![exp7_img1](https://github.com/user-attachments/assets/37a6b256-1021-4189-93c5-a979b9c18ae4)

![exp7_img2](https://github.com/user-attachments/assets/422a8c56-d074-4e8d-9638-4582bcd9f00d)


MEAN SQUARED ERROR

![exp7_img3](https://github.com/user-attachments/assets/8ab71919-0bac-4c90-a0cc-279b14e233a3)



FINIAL PREDICTION

![exp7_img4](https://github.com/user-attachments/assets/a5229581-7352-4385-a33b-8a1df8e17ebd)

## RESULT:
Thus we have successfully implemented the auto regression function using python.
