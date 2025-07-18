# CodSoft
Collection of Machine Learning Projects
This repository contains a collection of beginner-friendly machine learning projects, each focused on a classic dataset and a fundamental modelling technique.

1. Titanic Survival Prediction
Project Overview
This project focuses on building a model to predict whether a passenger on the Titanic survived or not. It's a classic binary classification problem that involves working with a dataset containing missing values and categorical features.

Dataset: Titanic-Dataset.csv

Model Used: A Logistic Regression model was used to classify passengers into 'Survived' or 'Did not Survive'.

Process:

Data Cleaning: Handled missing values in 'Age' and 'Embarked' columns and dropped the 'Cabin' column.

Feature Engineering: Converted categorical features like 'Sex' and 'Embarked' into numerical format using one-hot encoding.

Model Training: Trained a Logistic Regression classifier on the prepared data.

Evaluation: Assessed the model using accuracy, a classification report, and a confusion matrix.

2. Movie Rating Prediction
Project Overview
The goal of this project is to predict the rating of a movie based on features like its genre, director, and actors. This is a regression problem aimed at estimating a continuous value (the rating).

Dataset: IMDb Movies India.csv

Model Used: A Random Forest Regressor was used to predict movie ratings. This model is powerful for capturing complex relationships in the data.

Process:

Data Cleaning: Cleaned and converted 'Year' and 'Duration' columns to numerical types and simplified the 'Genre' feature.

Data Preparation: Used a pipeline to apply one-hot encoding to categorical features like 'Genre', 'Director', and 'Actors'.

Model Training: Trained the Random Forest model on the processed data.

Evaluation: Measured performance using regression metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and the R-squared (R2) score. The results were also visualized.

3. Iris Flower Classification
Project Overview
This project tackles the famous Iris flower classification problem. The objective is to develop a model that can accurately classify Iris flowers into three species (setosa, versicolor, and virginica) based on their sepal and petal measurements.

Dataset: IRIS.csv

Model Used: A Logistic Regression model was implemented for this multi-class classification task.

Process:

Data Exploration: Visualized the relationships between features using a pairplot to see how the species cluster.

Data Preparation: Split the data into training and testing sets.

Model Training: Trained the classifier to learn the patterns associated with each species.

Evaluation: Evaluated the model's accuracy and visualized its performance with a detailed confusion matrix.

4. Sales Prediction from Advertising
Project Overview
This project involves forecasting product sales based on advertising expenditure across different media channels (TV, Radio, and Newspaper). It's a regression task designed to help businesses optimize their advertising strategies.

Dataset: advertising.csv

Model Used: A Linear Regression model was used to understand the direct relationship between advertising spend and sales.

Process:

Data Exploration: Analysed the data and visualized the relationship between each advertising platform and sales.

Model Training: Trained a Linear Regression model to predict sales based on the advertising budget.

Evaluation: Assessed the model using the R-squared score to determine how well it explains the variance in sales.

Visualization: Created a scatter plot to compare the model's predicted sales against the actual sales figures.
