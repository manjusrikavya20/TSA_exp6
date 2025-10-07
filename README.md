# TSA_exp6
## Devloped by: MANJUSRI KAVYA R
## Register Number: 212224040186
## Date:06/10/25
## Ex.No: 6 HOLT WINTERS METHOD
## AIM:
To implement the Holt Winters Method Model using Python.

## ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model predictions against test data
6. Create the final model and predict future data and plot it

## PROGRAM:
```
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

# Load the dataset
file_path = "gld_price_data.csv"  # Update with your actual file path
data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

# Use only the GLD (Gold price) column
data['GLD'] = pd.to_numeric(data['GLD'], errors='coerce')
data = data.dropna(subset=['GLD'])

# Resample the data to monthly frequency (mean of each month)
monthly_data = data['GLD'].resample('MS').mean()

# Split the data into train and test sets (90% train, 10% test)
train_data = monthly_data[:int(0.9 * len(monthly_data))]
test_data = monthly_data[int(0.9 * len(monthly_data)):]

# Holt-Winters model with additive trend and seasonality
fitted_model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='add',
    seasonal_periods=12  # yearly seasonality for monthly data
).fit()

# Forecast on the test set
test_predictions = fitted_model.forecast(len(test_data))

# Plot the results
plt.figure(figsize=(12, 8))
train_data.plot(legend=True, label='Train')
test_data.plot(legend=True, label='Test')
test_predictions.plot(legend=True, label='Predicted')
plt.title('Train, Test, and Predicted using Holt-Winters (Additive Trend/Seasonality)')
plt.show()

# Evaluate model performance
mae = mean_absolute_error(test_data, test_predictions)
mse = mean_squared_error(test_data, test_predictions)
print(f"Mean Absolute Error = {mae:.4f}")
print(f"Mean Squared Error = {mse:.4f}")

# Fit the model to the entire dataset and forecast the future
final_model = ExponentialSmoothing(
    monthly_data,
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()

forecast_predictions = final_model.forecast(steps=12)  # Forecast 12 future months

# Plot the original and forecasted data
plt.figure(figsize=(12, 8))
monthly_data.plot(legend=True, label='Original Data')
forecast_predictions.plot(legend=True, label='Forecasted Data', color='red')
plt.title('Original and Forecasted Gold Prices (Holt-Winters Additive)')
plt.show()

```
## OUTPUT:
## TEST_PREDICTION

<img width="992" height="705" alt="image" src="https://github.com/user-attachments/assets/51508772-0dec-4393-be96-f06356eea8e6" />

## FINAL_PREDICTION

<img width="986" height="701" alt="image" src="https://github.com/user-attachments/assets/4bbaff4d-ae51-4b10-a0ae-66d34691aeb8" />

## RESULT:
Thus the program run successfully based on the Holt Winters Method model.
