# Project 2 MOVIE RATING PREDICTION WITH PYTHON
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Loading the Data 
try:
    df = pd.read_csv('IMDb Movies India.csv', encoding='ISO-8859-1')
    print("IMDb movie dataset loaded")
    print("A quick look at the data:")
    print(df.head())
except FileNotFoundError:
    print("Error can't find 'IMDb Movies India.csv'.")
    exit()

# 2. Data Cleaning
df.dropna(subset=['Rating', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Year'], inplace=True)
print("Removed movies with missing ratings or key features.")
df['Year'] = df['Year'].str.extract(r'(\d{4})').astype(int)
print("Cleaned up the 'Year' column.")
df['Duration'] = df['Duration'].str.replace(' min', '').astype(int)
print("Cleaned up the 'Duration' column.")
df['Genre'] = df['Genre'].apply(lambda x: x.split(',')[0])
print("Simplified the 'Genre' column.")


# 3. Building the model
features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Year', 'Duration']
target = 'Rating'
X = df[features]
y = df[target]
categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
numerical_features = ['Year', 'Duration']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])


# 4. Training the Model
print("\nSplitting data and training the model.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training the model on {X_train.shape[0]} movies.")
model_pipeline.fit(X_train, y_train)
print("Model training is complete")


# 5. Evaluating the Model
print("\nEvaluating model performance")
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# 6. Visualizing the Results
print("\nCreating a plot to show the results")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2, label='Perfect Prediction')
plt.xlabel("Actual Ratings", fontsize=12)
plt.ylabel("Predicted Ratings", fontsize=12)
plt.title("Actual vs. Predicted Movie Ratings", fontsize=14)
plt.legend()
plt.show()
print("\nMovie Rating Prediction With Python Finished")
