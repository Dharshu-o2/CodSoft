# Project 4 Sales Prediction Using Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Dataset 
try:
    df = pd.read_csv('advertising.csv')
    print("Advertising dataset loaded successfully!")
    print(df.head())
except FileNotFoundError:
    print("Error can't find 'advertising.csv'.")
    exit()

#2. Explore the Data
print("\nChecking for missing values:")
print(df.isnull().sum())
print("\nCreating a pairplot to see how variables relate")
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='reg')
plt.suptitle("Relationship between Advertising Spend and Sales", y=1.02)
plt.show()


#3. Preparing the Data for Modeling
features = ['TV', 'Radio', 'Newspaper']
X = df[features]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data has been split into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples.")


#4. Train the Regression Model
print("\nTraining the Linear Regression model")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training is completed")
print("\nModel Coefficients:")
print(pd.DataFrame(model.coef_, X.columns, columns=['Coefficient']))


#5. Model Evaluation
print("\nLet's see how our model performed")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")
print(f"An R2 score of {r2:.2f} means that our model can explain {r2:.2%} of the variance in sales.")


#6. Visualize the Predictions

print("\nCreating a plot to visualize actual vs. predicted sales")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2, label='Perfect Prediction')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual Sales vs. Predicted Sales")
plt.legend()
plt.grid(True)
plt.show()

print("\nSales Prediction Using Python Finished")
