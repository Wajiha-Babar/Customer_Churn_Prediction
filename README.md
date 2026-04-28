# 🏦 Customer Churn Prediction Dashboard

<p align="center">
  <b>Bank Customer Churn Prediction using Machine Learning</b><br>
  <i>EDA • Data Cleaning • Feature Engineering • Model Training • Feature Importance • Streamlit Dashboard</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-red?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/Visualization-Plotly%20%7C%20Seaborn-purple?style=for-the-badge">
</p>

---

## 👩‍💻 Author

**Wajiha Babar**  
Data Science & Machine Learning Project

---

## 📌 Project Overview

Customer churn is one of the most important business problems in the banking sector. Churn means a customer leaves the bank or stops using banking services.

The objective of this project is to build a machine learning solution that predicts whether a bank customer is likely to leave the bank.

This project includes:

- Complete data understanding
- Data cleaning and preparation
- Exploratory Data Analysis with professional graphs
- Categorical feature encoding
- Supervised classification model training
- Model evaluation using multiple metrics
- Feature importance analysis
- Premium interactive Streamlit dashboard
- Single customer churn prediction
- Batch prediction using uploaded CSV files

---

## 🎯 Task Objective

The main goal is to identify bank customers who are likely to churn.

| Target Value | Meaning |
|---|---|
| `Exited = 0` | Customer did not churn |
| `Exited = 1` | Customer churned |

---

## 📂 Dataset

This project uses a bank customer churn dataset containing customer demographic and banking behavior features.

### Main Files

```text
train.csv
test.csv
sample_submission.csv
```

### Important Columns

| Column | Description |
|---|---|
| `CreditScore` | Customer credit score |
| `Geography` | Customer country/location |
| `Gender` | Customer gender |
| `Age` | Customer age |
| `Tenure` | Number of years customer stayed with bank |
| `Balance` | Customer account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Whether customer has a credit card |
| `IsActiveMember` | Whether customer is active |
| `EstimatedSalary` | Estimated customer salary |
| `Exited` | Target column for churn |

---

## 🧠 Machine Learning Approach

The project follows a complete machine learning workflow:

1. Load the dataset
2. Understand dataset structure
3. Check missing values and duplicates
4. Clean and prepare the dataset
5. Perform EDA using graphs
6. Encode categorical columns using One-Hot Encoding
7. Scale numerical columns using StandardScaler
8. Train multiple classification models
9. Tune prediction threshold
10. Evaluate models using classification metrics
11. Select the best model using ROC-AUC score
12. Analyze feature importance
13. Save the trained model
14. Build an interactive Streamlit dashboard

---

## 🤖 Models Used

The following machine learning models were trained and compared:

| Model |
|---|
| Logistic Regression |
| Random Forest Classifier |
| Extra Trees Classifier |
| Hist Gradient Boosting Classifier |

The final selected model is:

```text
Hist Gradient Boosting Classifier
```

---

## 📈 Model Performance

Validation performance of the best model:

| Metric | Score |
|---|---:|
| Accuracy | 0.8565 |
| Precision | 0.6572 |
| Recall | 0.6725 |
| F1 Score | 0.6648 |
| ROC-AUC | 0.8896 |
| MAE Probability | 0.1966 |
| RMSE Probability | 0.3129 |
| Optimal Threshold | 0.34 |

The model was selected based on **ROC-AUC score**, because customer churn is an imbalanced classification problem.

---

## 📊 Exploratory Data Analysis

The notebook includes detailed EDA visuals:

- Customer churn distribution
- Churn percentage pie chart
- Churn rate by geography
- Churn rate by gender
- Age distribution by churn status
- Balance distribution by churn status
- Credit score distribution
- Churn rate by number of products
- Active vs inactive member churn rate
- Tenure vs churn rate
- Salary distribution
- Correlation heatmap

---

## 🖥️ Premium Streamlit Dashboard

The project includes a clean, professional, luxury-style interactive dashboard built with Streamlit.

### Dashboard Features

- Professional white, maroon, and gold UI
- Sidebar filters
- Executive KPI cards
- Churn distribution chart
- Geography-based churn analysis
- Customer behavior insights
- Model performance section
- Confusion matrix
- Feature importance chart
- Single customer churn prediction form
- Batch prediction using uploaded CSV
- Downloadable prediction results

---

## 📌 Dashboard Sections

### 1. Executive Overview

Shows high-level business KPIs:

- Total customers
- Churned customers
- Churn rate
- Average age
- Average balance

### 2. Customer Insights

Includes visual insights for:

- Age distribution
- Balance distribution
- Number of products
- Active member status

### 3. Predict Customer

Allows users to enter customer details and predict churn risk.

Prediction output includes:

- Churn probability
- High risk or low risk label
- Recommended business action

### 4. Model Performance

Shows:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix
- Feature importance

### 5. Batch Prediction

Allows uploading a CSV file and generates churn predictions for all customers.

---

## 📁 Project Folder Structure

```text
Customer_Churn_Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── processed/
│
├── notebooks/
│   └── 01_customer_churn_prediction.ipynb
│
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── eda_visuals.py
│   └── train_model.py
│
├── models/
│   └── churn_model.pkl
│
└── outputs/
    ├── metrics.json
    ├── feature_importance.csv
    ├── submission.csv
    ├── dashboard_batch_predictions.csv
    └── figures/
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Wajiha-Babar/Customer-Churn-Prediction-Dashboard.git
cd Customer-Churn-Prediction-Dashboard
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

Place dataset files inside:

```text
data/raw/
```

Required files:

```text
train.csv
test.csv
sample_submission.csv
```

Because dataset files can be large, they may not be included in the GitHub repository.

---

## 🚀 How to Run the Project

### 1. Generate EDA Visuals

```bash
python src/eda_visuals.py
```

### 2. Train the Machine Learning Model

```bash
python src/train_model.py
```

This will generate:

```text
models/churn_model.pkl
outputs/metrics.json
outputs/feature_importance.csv
outputs/submission.csv
```

### 3. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in the browser at:

```text
http://localhost:8501
```

---

## 📓 Jupyter Notebook

The notebook contains the complete project workflow:

```text
notebooks/01_customer_churn_prediction.ipynb
```

Notebook includes:

- Introduction and problem statement
- Dataset understanding
- Data cleaning
- EDA graphs
- Feature encoding
- Model training
- Model evaluation
- Feature importance
- Final conclusion

---

## 🧪 Evaluation Metrics

The model is evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Correct churn predictions out of predicted churn |
| Recall | Actual churn customers correctly identified |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Ability to separate churn and non-churn customers |
| Confusion Matrix | Classification result summary |
| MAE Probability | Error in predicted churn probability |
| RMSE Probability | Probability prediction error magnitude |

---

## 🔍 Feature Importance

Feature importance helps identify the main factors that influence customer churn.

Important factors include:

- Age
- Number of products
- Active membership status
- Balance
- Geography
- Credit score

These insights can help banks design better retention strategies.

---

## 💼 Business Recommendations

Based on churn analysis, the bank should:

1. Focus on inactive customers
2. Create loyalty offers for high-risk customers
3. Improve engagement for customers with high balance but low activity
4. Monitor customers with higher churn probability
5. Offer personalized banking products
6. Use dashboard insights for customer retention campaigns

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Programming | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Dashboard | Streamlit |
| Model Saving | Joblib |
| Development | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |

---

## 📌 Key Results

The project successfully built a machine learning model to predict customer churn.

Final model:

```text
Hist Gradient Boosting Classifier
```

Best ROC-AUC:

```text
0.8896
```

The dashboard makes the project business-friendly by allowing users to explore customer behavior and predict churn risk interactively.

---

## ✅ Conclusion

This project demonstrates an end-to-end machine learning solution for customer churn prediction in the banking sector.

It covers data cleaning, EDA, feature engineering, model training, evaluation, feature importance, and deployment through a premium Streamlit dashboard.

The final dashboard can help decision-makers identify high-risk customers and take action before customers leave the bank.

---

## 👩‍💻 Developed By

**Wajiha Babar**

<p align="center">
  <b>Customer Churn Prediction Dashboard</b><br>
  <i>Machine Learning • Data Science • Business Intelligence</i>
</p>