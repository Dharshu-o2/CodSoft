# Project 1 TITANIC SURVIVAL PREDICTION
# Importing the libraries required for project
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Getting the Data
try:
    df = pd.read_csv('Titanic-Dataset.csv')
    print("The Titanic dataset is loaded.")
    print("Looking at the first few rows:")
    print(df.head())
    print("\n Quick summary of the data:")
    df.info()
except FileNotFoundError:
    print("Error couldn't find 'Titanic-Dataset.csv'.")
    exit()

# 2. Clearing the data
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)
print(f"Filled in the missing 'Age' gaps with the median age: {median_age}")

mode_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_embarked, inplace=True)
print(f"Filled in the missing 'Embarked' spots with the most common one: {mode_embarked}")

df.drop('Cabin', axis=1, inplace=True)
print("The 'Cabin' column was a mess, so I dropped it.")
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
print("Dropped 'PassengerId', 'Name', and 'Ticket' since they are not useful for prediction.")

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print("\nChanged 'Sex' and 'Embarked' into numerical columns for easier use.")
print("displaying the columns:", df.columns)


# 3. Building the Model
# Separating the data into two parts
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using Logistic Regression model to train 
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Training complete")


# 4. Checking our Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Did not Survive', 'Survived'],
            yticklabels=['Did not Survive', 'Survived'])
plt.ylabel('Actually Happened Results')
plt.xlabel('Model Guessed Results')
plt.title('Confusion Matrix')
plt.show()

print("\n Titanic Survival Prediction Finished")
