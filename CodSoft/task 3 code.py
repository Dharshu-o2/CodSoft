#Project 3 Iris Flower Classification Model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Dataset
try:
    df = pd.read_csv('IRIS.csv')
    print("Iris dataset loaded successfully!")
    print("Here's a look at the first few rows:")
    print(df.head())
    print("\nLet's see how many flowers of each species we have:")
    print(df['species'].value_counts())
except FileNotFoundError:
    print("Error can't find 'IRIS.csv'.")
    exit()

# 2. Explore the Data
print("\nCreating a pairplot to visualize the data")
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.suptitle("Pairplot of Iris Data by Species", y=1.02)
plt.show()


# 3. Prepare the Data for Modeling
print("\nGetting the data ready for our model")
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into training and testing sets.")
print(f"Training set has {X_train.shape[0]} samples.")
print(f"Testing set has {X_test.shape[0]} samples.")


# 4. Train the Classification Model 
print("\nTraining our classification model")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print("Model training is complete!")


#5. Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


# 6. Visualize the Confusion Matrix
print("\nCreating a plot for the confusion matrix")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)

plt.ylabel('Actual Species')
plt.xlabel('Predicted Species')
plt.title('Confusion Matrix for Iris Classification')
plt.show()

print("\nIris Flower Classification Model Finished")
