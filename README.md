# Data Mining Homework 2

This repository contains the Python implementation and solutions for **Data Mining Homework 2**. The homework focuses on decision trees, dimensionality reduction, and performance evaluation using various datasets.

## Repository Structure

- **DataMining_HW2.pdf**: The original homework assignment PDF containing the problem statements and solutions.
- **HW2.1.py**: Python script for Problem 1, which involves loading the Iris dataset and inducing binary decision trees with different depths.
- **HW2.2.py**: Python script for Problem 2, which involves loading the Breast Cancer Wisconsin (Diagnostic) dataset and calculating entropy, Gini index, misclassification error, and information gain.
- **HW2.3.py**: Python script for Problem 3, which involves PCA dimensionality reduction on the Breast Cancer Wisconsin (Diagnostic) dataset and comparing model performance with and without PCA.

## Problem Descriptions

### Problem 1: Decision Trees on Iris Dataset
- **Objective**: Load the Iris dataset using `sklearn` and induce binary decision trees with varying maximum depths (`max_depth` from 1 to 5).
- **Tasks**:
  - Evaluate the model's performance using **Recall**, **Precision**, and **F1 Score**.
  - Analyze the impact of tree depth on model performance.
  - Explain the differences between **micro**, **macro**, and **weighted** methods for score calculation.

### Problem 2: Decision Trees on Breast Cancer Wisconsin Dataset (Discrete)
- **Objective**: Load the Breast Cancer Wisconsin (Diagnostic) dataset (discrete version) and induce a binary decision tree with a maximum depth of 2.
- **Tasks**:
  - Calculate **Entropy**, **Gini Index**, and **Misclassification Error** for the first split.
  - Determine the **Information Gain** and identify the feature selected for the first split.
  - Analyze the decision boundary and its impact on classification.

### Problem 3: PCA and Decision Trees on Breast Cancer Wisconsin Dataset (Continuous)
- **Objective**: Load the Breast Cancer Wisconsin (Diagnostic) dataset (continuous version) and perform PCA dimensionality reduction before inducing a binary decision tree.
- **Tasks**:
  - Compare the model's performance (F1 Score, Precision, Recall) using the original data versus PCA-reduced data (1 and 2 components).
  - Analyze the **Confusion Matrix** to determine **False Positives (FP)**, **True Positives (TP)**, **False Positive Rate (FPR)**, and **True Positive Rate (TPR)**.
  - Discuss the benefits of using continuous data and PCA dimensionality reduction.

## Requirements

To run the code in this repository, you will need the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn` (`sklearn`)
