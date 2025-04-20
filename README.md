# Failure-Detection
In this file you can find projects that apply machine learning to detect errors (using data frames from official websites).
# 🛠️ Predictive Maintenance: Machine Failure Detection Using Sensor Data
# 🛠️ Predictive Maintenance: Machine Failure Detection Using Sensor Data

This project focuses on building a machine learning model to **predict industrial machine failures** based on multivariate sensor readings. By leveraging supervised learning techniques, we aim to detect early warning signs of failure and support smarter maintenance decisions.

---

## 📂 Project Structure

- `notebooks/`: Jupyter notebooks with step-by-step analysis and modeling
- `data/`: Cleaned and preprocessed datasets
- `models/`: Trained models (e.g., `.pkl` files)
- `README.md`: Project overview
- `requirements.txt`: List of dependencies

---

## 🎯 Objectives

- Explore and clean real-world sensor data from industrial machines.
- Detect patterns and outliers in variables related to mechanical and environmental factors.
- Train classification models to predict machine failure events.
- Identify key variables that contribute to prediction accuracy.
- Provide actionable insights that can help optimize sensor placement and maintenance schedules.

---

## 📊 Dataset Overview

The dataset includes sensor readings such as:

- `footfall`: People or object traffic near the machine
- `VOC`: Volatile organic compound levels
- `AQ`: Air quality index
- `USS`: Ultrasonic sensor (proximity)
- `CS`: Current sensor
- `RP`: Rotational position (RPM)
- `IP`: Input pressure
- `Temperature` & `tempMode`
- `fail`: Target label (1 = failure, 0 = normal)

---

## 🧪 Methodology

### 🔍 Exploratory Data Analysis (EDA)
- Analyzed distributions and detected outliers using histograms and boxplots
- Examined variable correlations using heatmaps

### 🧹 Preprocessing
- Applied `log1p` transformation to reduce skewness in `footfall`
- Scaled numerical features using `RobustScaler` to minimize outlier impact
- Split data into 80% training and 20% testing using stratified sampling

### 🤖 Modeling
- Trained two models:
  - **Logistic Regression** – Fast and interpretable baseline
  - **Random Forest** – Ensemble method with higher accuracy
- Evaluated models using **Precision**, **Recall**, and **F1-score**
- Visualized ROC curves to assess classification performance

### 🌟 Feature Importance
- Extracted feature importances from the Random Forest model
- Top predictors: **VOC**, **AQ**, **USS**, and **CS**

---

## 📈 Results

| Model              | Precision | Recall | F1-score |
|-------------------|-----------|--------|----------|
| Logistic Regression | 88.8%    | 89.9%  | 89.3%    |
| Random Forest       | 90.1%    | 92.4%  | 91.3%    |

Random Forest outperformed Logistic Regression in all metrics, while remaining interpretable and robust.

---

## 🧠 Key Takeaways

- **Environmental features** (like VOC and AQ) can be strong predictors, even more than mechanical ones.
- Preprocessing plays a major role in model success.
- Even simple models can perform well with clean, scaled data.

---

## 🚀 Future Work

- Collect more data with **timestamps** to enable time-series modeling (e.g., LSTM).
- Expand dataset with multiple machine types and failure scenarios.
- Try advanced models like **XGBoost**, **LightGBM**, and **CatBoost**.
- Integrate with **real-time monitoring systems** and dashboards.
- Use explainability tools like **SHAP** and **LIME** for deeper model transparency.

---




