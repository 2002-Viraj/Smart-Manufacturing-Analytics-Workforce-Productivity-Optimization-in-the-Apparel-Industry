import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load data
df = pd.read_excel(
r"C:\Users\viraj\OneDrive\Desktop\Smart_Manufacturing_Analytics_Project\03_Python_Analysis\garments_worker_productivity(AutoRecovered).xlsx"
)

# Date format
df['date'] = pd.to_datetime(df['date'])

# Monthly productivity
monthly_prod = df.groupby(
    pd.Grouper(key='date', freq='ME')
)['actual_productivity'].mean()

# Model
model = ExponentialSmoothing(
    monthly_prod,
    trend='add',
    seasonal=None
)

fit = model.fit()

# Forecast only 24 months (2016-2017)
forecast = fit.forecast(24)

# Plot
plt.figure(figsize=(12,6))

plt.plot(
    monthly_prod.index,
    monthly_prod.values,
    color='blue',
    marker='o',
    linewidth=2,
    label='Historical Productivity'
)

plt.plot(
    forecast.index,
    forecast.values,
    color='red',
    marker='o',
    linewidth=3,
    label='Forecast Productivity'
)

plt.title(
    'Workforce Productivity Forecast (2016-2017)'
)

plt.xlabel('Year')
plt.ylabel('Average Productivity')

plt.legend()
plt.grid(True)

plt.show()