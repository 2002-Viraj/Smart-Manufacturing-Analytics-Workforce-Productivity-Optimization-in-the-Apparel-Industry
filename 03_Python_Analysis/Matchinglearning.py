import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_excel(
    r"C:\Users\viraj\OneDrive\Desktop\Smart_Manufacturing_Analytics_Project\03_Python_Analysis\garments_worker_productivity(AutoRecovered).xlsx"
)

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Column Names:")
print(df.columns.tolist())

X = df[['targeted_productivity',
        'over_time',
        'incentive',
        'idle_time',
        'no_of_workers']]

y = df['actual_productivity']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("R2 Score:", r2_score(y_test, predictions))
print("MAE:", mean_absolute_error(y_test, predictions))

print("\nFeature Importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")
    
    
    import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,5))

# Scatter Plot
plt.scatter(y_test, predictions, alpha=0.7)

# Trend Line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)

plt.plot(y_test, p(y_test), linewidth=2)

plt.xlabel("Actual Productivity")
plt.ylabel("Predicted Productivity")
plt.title("Actual vs Predicted Productivity")

plt.grid(True)

plt.show()


#-----------------------------------

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,5))

# Scatter Plot
plt.scatter(y_test, predictions, alpha=0.7)

# Red Trend Line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)

plt.plot(
    y_test,
    p(y_test),
    color='red',
    linewidth=3,
    label='Trend Line'
)

plt.xlabel("Actual Productivity")
plt.ylabel("Predicted Productivity")
plt.title("Actual vs Predicted Productivity")

plt.legend()
plt.grid(True)

plt.show()

#--------------
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.scatter(y_test, predictions, alpha=0.7)

# Perfect Prediction Line (Red)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linewidth=3,
    label='Perfect Prediction'
)

plt.xlabel("Actual Productivity")
plt.ylabel("Predicted Productivity")
plt.title("Actual vs Predicted Productivity")

plt.legend()
plt.grid(True)

plt.show()

#bargrh-------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Productivity")
plt.ylabel("Predicted Productivity")

plt.title("Actual vs Predicted Productivity")

plt.show()

import matplotlib.pyplot as plt

features = ['targeted_productivity',
            'over_time',
            'incentive',
            'idle_time',
            'no_of_workers']

importance = [0.3361, 0.2116, 0.1914, 0.0266, 0.2342]

plt.figure(figsize=(8,5))
plt.bar(features, importance)

plt.title("Feature Importance")
plt.xlabel("Variables")
plt.ylabel("Importance Score")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()