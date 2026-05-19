# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 19/05/2026

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_excel('/content/advanced_sales_report.xlsx')

# Convert Year column to datetime format
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

# Set Year column as index
data.set_index('Year', inplace=True)

# Plot Accuracy Time Series
plt.plot(data.index, data['Accuracy'])

plt.xlabel('Year')
plt.ylabel('Accuracy')

plt.title('Accuracy Time Series')

plt.grid(True)

plt.show()

# Function to check stationarity
def check_stationarity(timeseries):

    # Perform Dickey-Fuller Test
    result = adfuller(timeseries)

    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

    print('Critical Values:')

    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

# Check stationarity
check_stationarity(data['Accuracy'])

# Plot ACF
plot_acf(data['Accuracy'])

plt.show()

# Plot PACF
plot_pacf(data['Accuracy'])

plt.show()

# Build SARIMA model
sarima_model = SARIMAX(
    data['Accuracy'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 2)
)

sarima_result = sarima_model.fit()

# Split train and test data
train_size = int(len(data) * 0.8)

train = data['Accuracy'][:train_size]
test = data['Accuracy'][train_size:]

# Train SARIMA model
sarima_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 2)
)

sarima_result = sarima_model.fit()

# Predictions
predictions = sarima_result.predict(
    start=len(train),
    end=len(train) + len(test) - 1
)

# RMSE Calculation
mse = mean_squared_error(test, predictions)

rmse = np.sqrt(mse)

print('RMSE:', rmse)

# Plot Actual vs Predicted
plt.plot(test.index, test, label='Actual')

plt.plot(test.index,
         predictions,
         color='red',
         label='Predicted')

plt.xlabel('Year')
plt.ylabel('Accuracy')

plt.title('SARIMA Model Predictions')

plt.legend()

plt.grid(True)

plt.show()
```
### OUTPUT:

<img width="388" height="290" alt="image" src="https://github.com/user-attachments/assets/858da9b9-df0a-4465-b662-a49c7aa3dcf6" />


### RESULT:
Thus the program run successfully based on the SARIMA model.
