# Customer Churn Analysis — End-to-End Data Project

**Author:** Yatharth Sharma

**Created:** December 2025  

**Tools Used:** Python, MySQL, Power BI, Pandas, Scikit-Learn, XGBoost  

---

Project Overview

This project analyzes customer churn for a telecom company using an end-to-end data workflow:

1. Raw dataset (Kaggle)  
2. Data cleaning and transformation in MySQL  
3. Exploratory Data Analysis (EDA) and Machine Learning in Python  
4. Interactive visual analytics dashboard in Power BI  
5. Churn probability prediction using Logistic Regression and XGBoost  

The goal is to identify drivers of churn, quantify customer risk, and provide actionable insights for retention.

---

Repository Structure

data/ 

raw/ → raw Kaggle dataset  
cleaned/ → cleaned dataset exported from SQL  
sql/ → SQL transformation script  
python/ → EDA + ML pipeline  
powerbi/ → PBIX file + screenshots  
outputs/ → churn predictions from XGBoost  
README.md → documentation  

---

1. Data Preparation (MySQL)

Key steps performed in SQL:

- Handling missing values  
- Creating tenure groups  
- Converting data types  
- Encoding categorical variables  
- Removing duplicates  
- Exporting final clean dataset  

---

2. Exploratory Data Analysis (Python)

Using pandas, seaborn, and matplotlib, we explored:

- Churn rate by contract type  
- Churn by internet service  
- Monthly charges distribution  
- Churn by tenure buckets  
- Gender, senior citizen, partner impact  
- Revenue and services comparison  

---

3. Machine Learning Models

**Models Used:**  
- Logistic Regression (baseline)  
- XGBoost Classifier (final model)

**Performance (ROC-AUC):**

Model | ROC-AUC  
------|---------  
Logistic Regression | ~0.78  
XGBoost | ~0.85  

---

4. Power BI Dashboard

The dashboard includes:

**KPIs:**  
- Total Customers  
- Churn Rate %  
- Churned Customers  
- Avg Monthly Charges  
- Avg Tenure  

**Insights Tabs:**  
- KPI Overview  
- Customer Segmentation  
- Services & Revenue  

---

5. Key Insights & Recommendations

### Insights
- Month-to-month customers have 4× higher churn  
- Fiber optic users show highest churn among internet customers  
- Tenure < 12 months has the highest churn rate  
- Senior citizens with no partner leave at extremely high rates  
- Electronic check payments have the highest churn → billing friction  
- Customers with 0 or 1 services churn more often  

### Recommendations
- Offer incentives for 1-year/2-year contracts  
- Improve fiber-optic customer experience (service quality)  
- Early-stage onboarding and retention for new customers (< 6 months)  
- Simplify or discourage electronic check payments  
- Bundle more services → reduce churn  

---

Conclusion

This end-to-end churn project shows how combining SQL, Python, ML, and Power BI provides deep customer insights and actionable business strategies.

---

Future Work

- Deploy churn prediction as an API  
- Build automated ETL pipeline  
- Add SHAP model explainability  

