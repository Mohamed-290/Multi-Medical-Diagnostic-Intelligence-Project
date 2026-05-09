# Multi-Medical Diagnostic Intelligence 🩺🤖
Multi-Medical Diagnostic Intelligence Project

## Overview

Multi-Medical Diagnostic Intelligence is an AI-powered healthcare project designed to assist in early disease prediction and medical diagnosis using Machine Learning and Deep Learning techniques. The system integrates multiple predictive models into one intelligent platform capable of analyzing medical data and MRI images to support clinical decision-making.

This project focuses on four major medical applications:

* Diabetes Disease Prediction
* Heart Disease Prediction
* Chronic Kidney Disease Prediction
* Brain Tumor Detection using MRI Images

The project combines data preprocessing, visualization, machine learning classification, deep learning, and deployment technologies to create an interactive diagnostic system.

---

# Project Objectives

* Build intelligent predictive models for multiple diseases.
* Improve early diagnosis accuracy using AI techniques.
* Compare different machine learning algorithms.
* Apply Deep Learning for medical image classification.
* Deploy the models using Streamlit for user interaction.

---

# Technologies & Libraries Used

## Programming Language

* Python

## Data Analysis & Visualization

* Pandas
* NumPy
* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn
* XGBoost

## Deep Learning

* TensorFlow
* Keras
* OpenCV

## Deployment

* Streamlit
* Joblib

---

# Modules Included

## 1. Diabetes Disease Prediction

Machine learning classification model used to predict diabetes based on medical measurements.

### Algorithms Tested

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* XGBoost

### Best Model

* Support Vector Machine (SVM)
* Accuracy: 90.78%

---

## 2. Heart Disease Prediction

Predicting heart disease risk using clinical and health-related attributes.

### Algorithms Tested

* Logistic Regression
* KNN
* Gradient Boosting
* Decision Tree
* Random Forest
* XGBoost

### Best Model

* Gradient Boosting
* Accuracy: 83.51%

---

## 3. Chronic Kidney Disease Prediction

Classification model for detecting chronic kidney disease using clinical indicators.

### Algorithms Tested

* Logistic Regression
* KNN
* Gradient Boosting
* Decision Tree
* Random Forest
* SVM

### Best Model

* Decision Tree
* Accuracy: 98.75%

---

## 4. Brain Tumor Detection

Deep Learning model using Convolutional Neural Networks (CNN) for MRI image classification.

### CNN Workflow

* Image preprocessing
* Data augmentation
* Convolution + MaxPooling layers
* Flattening layer
* Dense layers with Dropout
* Softmax activation

### Features

* MRI image classification
* Tumor vs Non-Tumor prediction
* Regularization using Dropout
* Batch normalization for stable training

---

# Data Processing Workflow

* Data Cleaning
* Missing Value Handling
* Outlier Removal
* Data Encoding
* Feature Scaling & Normalization
* Exploratory Data Analysis (EDA)
* Correlation Heatmaps
* Model Evaluation

---

# Deployment

The trained models were saved using Joblib and deployed through a Streamlit web application to provide a simple and interactive user interface for predictions.

-----------------------------------------------------------------------------
## 📦 Data Sources (Datasets)

### Brain Tumor Images

The Brain Tumor image dataset has been uploaded to **Kaggle** due to its large size.

🔗 **Kaggle Dataset Link:**  
(https://www.kaggle.com/datasets/mohamedaymanhassan/brain-tumor-images-classification-by-cnn)

> ⚠️ **Note:** These images are for research purposes only and are not a substitute for medical diagnosis.

------------------------------------------------------
## ⚕️ Medical Disclaimer

This project is for **research and educational purposes only**.  
Do not use for actual medical diagnosis without consulting a qualified physician.
----------------------------------------------------------------
