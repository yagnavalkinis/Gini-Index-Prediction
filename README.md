# Gini-Index-Analysis
This repository contains code for analyzing Gini Index using machine learning techniques. The code is implemented in a Jupyter notebook (`Gini_Index.ipynb`) using Python and various libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Correlation Analysis](#correlation-analysis)
- [Handling Outliers](#handling-outliers)
- [Model Training](#model-training)
- [Data Scaling](#data-scaling)
- [Decision Tree Regression](#decision-tree-regression)
- [Random Forest Regression](#random-forest-regression)
- [Prediction](#prediction)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
The goal of this project is to analyze and predict the Gini Index based on various socio-economic indicators. The analysis includes data preprocessing, exploratory data analysis, correlation analysis, handling outliers, and training machine learning models.

## Project Structure
- `Gini_Index.ipynb`: Jupyter notebook containing the code for Gini Index analysis.
- `Dataset.csv`: Original dataset in CSV format.
- `README.md`: Project documentation.

## Setup
To run the code, you can use a Jupyter notebook environment such as Google Colab. The notebook is self-contained, and the required libraries will be installed as needed.

## Dataset
The dataset (`Dataset.xlsx`) used in this analysis contains socio-economic indicators over multiple years. The data is loaded, and irrelevant attributes are removed during preprocessing.

## Data Preprocessing
Data preprocessing involves tasks such as handling null values, renaming attributes, exploring data distribution, and removing unnecessary attributes.

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis involves univariate and bivariate analysis to understand the distribution and relationships between variables.

## Correlation Analysis
Correlation analysis is performed using a correlation matrix and visualizations to understand the relationships between different attributes.

## Handling Outliers
Outliers in certain attributes are handled using the Interquartile Range (IQR) method to improve the robustness of the models.

## Model Training
Machine learning models, including Decision Tree Regression and Random Forest Regression, are trained to predict the Gini Index based on the given features.

## Data Scaling
Data scaling techniques such as Min-Max Scaling and Standardization are applied to normalize the features.

## Decision Tree Regression
A Decision Tree Regression model is trained and evaluated on the dataset.

## Random Forest Regression
A Random Forest Regression model is trained and evaluated on the dataset.

## Prediction
The trained models are used to make predictions on new data.

## Conclusion
The analysis provides insights into the factors influencing the Gini Index and demonstrates the predictive capabilities of the trained models.

## References
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)

