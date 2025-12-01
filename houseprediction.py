import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression , LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score


df = pd.read_csv("house_prices_dataset.csv")
print(df.head())
print(df.columns)
print(df.shape)
print(df.describe())
print(df.info())


X = df[["square_feet", "num_rooms", "age", "distance_to_city(km)"]]
y = df["price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train_scaled, y_train)


square_feet = float(input("Enter House Area in sqft(500-3960): "))
if not (500 <= square_feet <= 3960):
    print("Invalid input! Please enter within the given range.")
    exit()




num_rooms = int(input("Enter num of rooms in your House(2-7): "))
if not (2 <= num_rooms <= 7):
    print("Invalid input! Please enter within the given range.")
    exit()




age = float(input("Enter age of House(0-99): "))
if not (0 <= age <= 99):
    print("Invalid input! Please enter within the given range.")
    exit()




distance = float(input("Enter your house distance from city (1-30km): "))
if not (1 <= distance <= 30):
    print("Invalid input! Please enter within the given range.")
    exit()



user_input = np.array([[square_feet, num_rooms, age, distance]])
user_input_scaled = scaler.transform(user_input)


predicted_price = model.predict(user_input_scaled)
print(f"According to your given information, your House Price will be {predicted_price[0]:.2f}")

y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test,y_pred)


mse = mean_squared_error(y_test , y_pred)
rmse = np.sqrt(mse)


print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")