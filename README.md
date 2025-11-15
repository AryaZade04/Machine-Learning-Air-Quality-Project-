# 🌫️ Air Quality Machine Learning Project  
*A Mini Project by Arya*

This mini project focuses on analyzing and predicting air-quality values using machine learning techniques.  
The dataset contains cleaned measurements from various locations along with timestamps and pollutant indicators.  
The goal is to build predictive models, explore hidden patterns using clustering, and reduce feature dimensions using PCA.

---

## 📌 Project Objectives
- Load and preprocess the dataset  
- Handle missing values using mean imputation  
- Encode categorical columns using Label Encoding  
- Train multiple supervised machine learning models  
- Apply PCA for dimensionality reduction  
- Perform K-Means clustering  
- Evaluate models based on performance metrics  
- Visualize PCA components and clusters  

---

## 📁 Dataset Information

- **File:** `Air_Quality_Cleaned_Data.csv`  
- **Total Rows:** 18,862  
- **Total Columns:** 10  

### **Key Variables**
- Name  
- Measure (pollutant / air-quality measure code)  
- Geo Place Name (state-level region)  
- Time Period  
- Start_Date  
- Data Value  
- Message  
- Year  
- Month  
- Day  

---

## 🛠 Machine Learning Models Used

### ✔ Supervised Learning
- Linear Regression  
- Decision Tree  
- Random Forest  
- XGBoost  

### ✔ Unsupervised Learning
- PCA (Principal Component Analysis)  
- K-Means Clustering  

These models were selected based on dataset characteristics and produced strong, interpretable results.

---

## 📊 Results Summary

| Model | Performance | Result |
|-------|-------------|--------|
| **Linear Regression** | Good | Conclusive |
| **Decision Tree** | Average | Partially Conclusive |
| **Random Forest** | Good | Conclusive |
| **XGBoost** | Good | Conclusive |
| **PCA** | — | Conclusive |
| **K-Means** | — | Partially Conclusive |

Random Forest and XGBoost delivered the best prediction performance.

---

## 🧩 Features Implemented
- Handling missing values  
- Label encoding of text columns  
- Data normalization (where required)  
- Model evaluation using R² score  
- PCA for dimensionality reduction  
- K-Means clustering visualization  

