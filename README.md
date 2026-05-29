# Fitness Progress Prediction & Analytics Dashboard

## Overview
An end-to-end predictive machine learning pipeline designed to forecast fitness progress and caloric expenditure based on physiological biometrics. Engineered on a 2,500+ record dataset, the system bridges the gap between raw health metrics and actionable foresight[cite: 1, 2, 3]. The baseline Multiple Linear Regression model achieved highly optimal accuracy with an R² score of 0.9803 and a Mean Absolute Error (MAE) of 30.27, outperforming non-linear ensemble methods on fundamental biological data[cite: 1, 2, 3].

The predictive pipeline is deployed as a real-time web application using the Streamlit framework, featuring a responsive "Bento Box" UI and dynamic Plotly visual analytics[cite: 3].

## Model Architecture & Mathematical Approach

To establish a robust predictive pipeline, two distinct machine learning architectures were evaluated to contrast linear scaling against non-linear decision boundaries[cite: 3]:

*   **Baseline Architecture (Multiple Linear Regression):** Models the relationship between the scalar response and multiple explanatory variables[cite: 3]. The model optimizes learned weights ($\theta_j$) by minimizing the Ordinary Least Squares (OLS) cost function[cite: 3]:
    $$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2$$
*   **Advanced Architecture (Random Forest Regressor):** An ensemble learning method initialized with 100 parallel decision trees to account for potential multi-feature interactions, outputting the unweighted average over all trees to mitigate high variance[cite: 3]:
    $$\hat{y}_{RF} = \frac{1}{B} \sum_{b=1}^B f_b(x)$$

*Conclusion: Linear Regression provided superior predictive accuracy while offering the "Explainable AI" (XAI) transparency required for a consumer health application, validating the hypothesis that physiological inputs scale linearly with energy expenditure[cite: 3].*

## Core Data Engineering & Features

*   **Exploratory Data Analysis (EDA):** Mapped exact mathematical pull via Pearson correlation coefficients, identifying Session Duration (0.91) and Average BPM (0.54) as the heaviest predictive weights[cite: 3].
*   **Advanced Data Sanitization:** Deployed complex Regular Expressions (Regex) and string-stripping methodologies to resolve "Dirty Data" anomalies (hidden newline/tab characters)[cite: 3].
*   **Feature Engineering:** Implemented One-Hot Encoding to transform nominal categorical variables into binary vectors and utilized the Interquartile Range (IQR) method for statistical outlier detection[cite: 3].
*   **Rule-Based Expert System:** A custom logic engine captures the numerical model output and cross-references user BMI, age, and workout modality to generate context-aware diet matrices and supplement stacks[cite: 3].
*   **Hard-Coded Safety Guardrails:** The Expert System includes strict conditional logic to prevent dangerous health recommendations, such as suggesting caloric surpluses for overweight individuals[cite: 3].

## Tech Stack
*   **Languages:** Python 
*   **Machine Learning:** Scikit-Learn, Pandas, NumPy
*   **Data Visualization:** Matplotlib, Plotly (Dashboard Analytics)
*   **Deployment:** Streamlit
*   **Environment:** Jupyter Notebooks

## Installation & Local Deployment

1. **Clone the repository:**
```bash
   git clone [https://github.com/Namenotvisible/fitness-progress-prediction.git](https://github.com/Namenotvisible/fitness-progress-prediction.git)
   cd fitness-progress-prediction
