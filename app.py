#predicting Household Electricity Consumption (mini version of grid-scale demand forcasting)
#import libraries & load datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

#Load a smaller CSV (hourly electricity usage)
#Example structure : "hour,consumption"
df = pd.read_csv("household_power_consumption.csv")
print(df.head())
#Select feature and target
#hour of the day
#usage of the day
# Create simple feature: previous hour's consumption
# df['hour']=df['consumption'].shift(1)
# Drop first row (since prev_hours is NaN there)
df = df.dropna()

X = df[['hour']]
y = df['consumption']

# split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4 : Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#evaluation model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error",mae)
print("R^2 Score",r2)

# Visualise predictions
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Electricity Consumption: Actual vs Predicted")
plt.show()
