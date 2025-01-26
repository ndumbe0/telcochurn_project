# TELCO-CHURN  PROJECT

## Table of Contents
1. **Introduction**
   - 1a. Needs of the Project
   - 1b. What I Need to Get Started
   - 1c. What Hypothesis I Am Testing
2. **Descriptive Analysis and EDA (Exploratory Data Analysis)**
   - Includes images for visualization
3. **Machine Learning Data Cleaning Process and Data Preparation for the Models**
   - 3a. Models Used on the Data and the Preferred Selected Model
4. **Churn Customer Prediction Model**
5. **Lifetime Value of Customer**
6. **Churn Rate**
7. **Conclusion**

---

## 1. Introduction

### 1a. Needs of the Project
The project aims to classify customers who are likely to churn based on their demographics, account information, and usage data.

### 1b. What I Need to Get Started
- **Python Libraries**: pandas, numpy, matplotlib, seaborn, plotly, sklearn, tkinter, flask.
- **Data Source**: Connected to a SQL server database containing customer churn data.

###Getting Started

Raw Data is being kept here:
dap-projects-database.database.windows.net
https://azubiafrica-my.sharepoint.com/personal/teachops_azubiafrica_org/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fteachops%5Fazubiafrica%5Forg%2FDocuments%2FCareer%20Accelerator%20Data%5FSets%2FLP2%20Datasets&ga=1
https://github.com/Azubi-Africa/Career_Accelerator_LP2-Classifcation/blob/main/LP2_Telco-churn-second-2000.csv
Data processing/transformation scripts are being kept (https://github.com/Azubi-Africa/Career_Accelerator_LP2-Classifcation/tree/main)


### 1c. What Hypothesis I Am Testing
Hypothesis: Will Customers with longer tenure are less likely to churn.
Null Hypothesis (
H0 ): 

Tenure has no impact on churn.

Alternative Hypothesis (Ha): 

Longer tenure reduces the likelihood of churn.


---

## 2. Descriptive Analysis and EDA (Exploratory Data Analysis)
Exploratory data analysis was conducted to understand the dataset's structure and identify patterns or anomalies.



### Key Insights:
- Distribution of churn rates across customer demographics.
- Correlation between tenure and churn likelihood.
- Payment methods and their impact on churn.

**Sample Visualization**:
![image](https://github.com/user-attachments/assets/b06d4459-3b5d-45e8-8813-48754cb082e8)
The histogram shows the distribution of Monthly Charges across customers. It appears that Monthly Charges range broadly, with a slight right skew, meaning there are more customers with lower charges compared to higher ones.

Most customers are clustered around the lower to mid-range of Monthly Charges, with fewer in the higher brackets.
![image](https://github.com/user-attachments/assets/9c4d6d97-e3a0-48de-ab74-5a7559c27c27)
The boxplot in the image compares Monthly Charges across different churn categories ("False", "True", "No", and "Yes"). Here's a breakdown of what the graph conveys:

It appears that "False" and "No" represent customers who did not churn, while "True" and "Yes" represent customers who churned. These categories might be duplicates or differently labeled groups.

For customers who did not churn ("False" or "No"), the median monthly charges are lower compared to those who churned ("True" or "Yes").
Customers who churned ("True" or "Yes") tend to have higher monthly charges, as indicated by the higher medians and interquartile ranges (IQRs).


![image](https://github.com/user-attachments/assets/633a1602-1dc8-4f27-aafd-dc8d95793b90)

**Churn by Gender (Bar Chart):**
The number of churned and non-churned customers is similar across genders.
Gender does not appear to have a significant impact on churn, as the proportions of churned customers are comparable for males and females.

![image](https://github.com/user-attachments/assets/39a9c9f8-cbf4-450e-834f-863016cf4145)

**Churn by Contract Type (Bar Chart):**
Customers with month-to-month contracts have a significantly higher churn rate compared to those with one-year or two-year contracts.
Customers on long-term contracts (one-year or two-year) are more likely to stay, suggesting that contract type is a strong predictor of churn.

![image](https://github.com/user-attachments/assets/77d485be-fdc5-4dd9-a18d-ff36ac01220b)

**Monthly Charges by Contract Type**
The box plot illustrates that monthly charges are slightly higher for customers on one-year contracts compared to those on month-to-month or two-year contracts. However, the distribution of charges is similar across all contract types, with no extreme outliers.

![image](https://github.com/user-attachments/assets/fb14c8af-3efa-417c-b924-ba71b6403e85)

**Total Charges vs. Tenure**
The scatter plot shows a positive correlation between tenure and total charges, as expected. Customers with longer tenures tend to have higher total charges, reflecting their extended engagement with the service.

---

## 3. Machine Learning Data Cleaning Process and Data Preparation for the Models

### Data Cleaning Steps:
- Handled missing values in columns like `TotalCharges`.
- Encoded categorical variables such as `InternetService`, `Contract`, etc.
- Scaled numerical features like `MonthlyCharges` and `tenure`.

### 3a. Models Used on the Data and the Preferred Selected Model
Several machine learning models were applied:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Machines (Preferred Model)

The Gradient Boosting Model was selected due to its superior accuracy and precision in predicting churn.

---

## 4. Churn Customer Prediction Model
The final model predicts whether a customer is likely to churn based on their account and usage data.

**Model Performance Metrics**:
- Accuracy: 85%
- Precision: 83%
- Recall: 78%

---

## 5. Lifetime Value of Customer
The lifetime value of a customer was calculated using their tenure, monthly charges, and contract type.

Formula:
\[
\text{Lifetime Value} = \text{Monthly Charges} \times \text{Tenure}
\]

---

## 6. Churn Rate
The overall churn rate was calculated as:
\[
\text{Churn Rate} = \frac{\text{Number of Churned Customers}}{\text{Total Customers}}
\]

---

## Conclusion
This project successfully developed a machine learning pipeline to predict customer churn with high accuracy. The insights from this analysis can be used to implement targeted retention strategies for at-risk customers.

---
FEATURED

https://dev.to/ndumbe0/telco-churn-classification-project-3pol

Owner (ndumbemoses@gmail.com) : [Moses N Ndumbe]

Team Leads (portia.bentum@azubiafrica.org) : [Ms.Portia Bentum]


