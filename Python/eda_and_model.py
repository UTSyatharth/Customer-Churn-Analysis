# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# ------------------------------------------------------------
# 1. CONNECT TO MYSQL & LOAD DATA
# ------------------------------------------------------------
print("🔗 Connecting to MySQL...")

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Pawan@0183",  # <- change this
    database="churn_db"
)

query = "SELECT * FROM customers_clean;"
df = pd.read_sql(query, connection)
connection.close()

print("✅ Data loaded successfully!")

print("\n🔹 Columns in df:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 2. BASIC EDA CHECKS
# ------------------------------------------------------------
print("\n🔹 Shape of data:", df.shape)

print("\n🔹 First 5 rows:")
print(df.head())

print("\n🔹 Data types:")
print(df.dtypes)

print("\n🔹 Missing values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 3. TARGET VARIABLE DISTRIBUTION
# ------------------------------------------------------------
print("\n🔹 Churn value counts:")
print(df['churn'].value_counts())

churn_rate = df['churn'].mean() * 100
print("\n🔹 Churn rate:", round(churn_rate, 2), "%")


# ------------------------------------------------------------
# 4. BASIC NUMERIC SUMMARIES
# ------------------------------------------------------------
print("\n🔹 Monthly charges summary:")
print(df['monthly_charges'].describe())

print("\n🔹 Tenure summary:")
print(df['tenure_months'].describe())

print("\n🔹 Total charges summary:")
print(df['total_charges'].describe())


# ------------------------------------------------------------
# 5. CATEGORICAL DISTRIBUTIONS
# ------------------------------------------------------------
cat_cols = ['gender', 'Contract', 'InternetService', 'PaymentMethod', 'tenure_group']

for col in cat_cols:
    if col in df.columns:
        print(f"\n🔹 Value counts for {col}:")
        print(df[col].value_counts())
    else:
        print(f"\n⚠️ Column {col} not found in dataframe!")


# ------------------------------------------------------------
# 6. VISUAL EDA
# ------------------------------------------------------------
plt.rcParams['figure.figsize'] = (8, 5)

# A. Churn count
sns.countplot(x='churn', data=df)
plt.title("Churn Count")
plt.tight_layout()
plt.show()

# B. Monthly charges vs churn
sns.boxplot(x='churn', y='monthly_charges', data=df)
plt.title("Monthly Charges by Churn")
plt.tight_layout()
plt.show()

# C. Tenure distribution
sns.histplot(df['tenure_months'], bins=30, kde=True)
plt.title("Tenure Distribution")
plt.tight_layout()
plt.show()

# D. Churn rate by Contract
if 'Contract' in df.columns:
    churn_by_contract = df.groupby('Contract')['churn'].mean().reset_index()
    churn_by_contract['churn_pct'] = churn_by_contract['churn'] * 100
    print("\n🔹 Churn rate by Contract:")
    print(churn_by_contract)

    sns.barplot(x='Contract', y='churn_pct', data=churn_by_contract)
    plt.title("Churn % by Contract Type")
    plt.tight_layout()
    plt.show()

# E. Churn vs Payment Method
if 'PaymentMethod' in df.columns:
    churn_by_payment = df.groupby('PaymentMethod')['churn'].mean().reset_index()
    churn_by_payment['churn_pct'] = churn_by_payment['churn'] * 100
    print("\n🔹 Churn rate by Payment Method:")
    print(churn_by_payment.sort_values('churn_pct', ascending=False))

    sns.barplot(x='churn_pct', y='PaymentMethod', data=churn_by_payment)
    plt.title("Churn % by Payment Method")
    plt.tight_layout()
    plt.show()

# F. Churn vs InternetService
if 'InternetService' in df.columns:
    churn_by_internet = df.groupby('InternetService')['churn'].mean().reset_index()
    churn_by_internet['churn_pct'] = churn_by_internet['churn'] * 100
    print("\n🔹 Churn rate by InternetService:")
    print(churn_by_internet.sort_values('churn_pct', ascending=False))

    sns.barplot(x='InternetService', y='churn_pct', data=churn_by_internet)
    plt.title("Churn % by Internet Service Type")
    plt.tight_layout()
    plt.show()

# G. Churn vs Tenure Group
if 'tenure_group' in df.columns:
    churn_by_tenure_group = df.groupby('tenure_group')['churn'].mean().reset_index()
    churn_by_tenure_group['churn_pct'] = churn_by_tenure_group['churn'] * 100
    print("\n🔹 Churn rate by Tenure Group:")
    print(churn_by_tenure_group)

    sns.barplot(x='tenure_group', y='churn_pct', data=churn_by_tenure_group)
    plt.title("Churn % by Tenure Group")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 6.5 EXPORT CLEAN DATA FOR POWER BI
# ------------------------------------------------------------
df.to_csv("customers_clean.csv", index=False)
print("\n✅ Clean dataset exported to customers_clean.csv for Power BI.")


# ------------------------------------------------------------
# 7. MODELING (LOGISTIC REGRESSION + XGBOOST)
# ------------------------------------------------------------
feature_cols = [
    'senior_citizen', 'partner', 'dependents', 'tenure_months',
    'monthly_charges', 'total_charges',
    'has_multiple_services', 'is_long_term_contract'
]

cat_features = ['InternetService', 'Contract', 'PaymentMethod']

# Build list of columns for modelling, include customerID if exists
cols_for_model = feature_cols + cat_features + ['churn']
if 'customerID' in df.columns:
    cols_for_model.append('customerID')

# Keep only columns that actually exist (safety)
cols_for_model = [c for c in cols_for_model if c in df.columns]

df_model = df[cols_for_model].dropna()

# Separate customerID if present
if 'customerID' in df_model.columns:
    customer_ids = df_model['customerID'].copy()
    df_model = df_model.drop(columns=['customerID'])
else:
    customer_ids = None

# One-hot encode categorical variables
df_model = pd.get_dummies(df_model, columns=[c for c in cat_features if c in df_model.columns],
                          drop_first=True)

X = df_model.drop('churn', axis=1)
y = df_model['churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 7.2 Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

y_pred_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
roc_lr = roc_auc_score(y_test, y_pred_proba_lr)
print("\n🔹 Logistic Regression ROC-AUC:", round(roc_lr, 3))

y_pred_lr = (y_pred_proba_lr >= 0.5).astype(int)
print("\n🔹 Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))


# 7.3 XGBoost (tree model – no scaling required)
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)
y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]
roc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
print("🔹 XGBoost ROC-AUC:", round(roc_xgb, 3))

y_pred_xgb = (y_pred_proba_xgb >= 0.5).astype(int)
print("\n🔹 XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))


# ------------------------------------------------------------
# 8. FINAL CHURN PREDICTIONS EXPORT
# ------------------------------------------------------------
# Retrain XGBoost on full dataset
xgb.fit(X, y)
churn_proba_all = xgb.predict_proba(X)[:, 1]

df_results = pd.DataFrame(index=X.index)
df_results['churn_probability'] = churn_proba_all


def risk_segment(p):
    if p >= 0.7:
        return "High"
    elif p >= 0.3:
        return "Medium"
    else:
        return "Low"


df_results['risk_segment'] = df_results['churn_probability'].apply(risk_segment)

# revenue_at_risk using monthly_charges from X (still a column)
if 'monthly_charges' in X.columns:
    df_results['revenue_at_risk'] = df_results['churn_probability'] * X['monthly_charges']

# Attach customerID if we had it
if customer_ids is not None:
    df_results['customerID'] = customer_ids.values

# Order columns nicely if customerID exists
cols_out = []
if 'customerID' in df_results.columns:
    cols_out.append('customerID')
cols_out += ['churn_probability', 'risk_segment']
if 'revenue_at_risk' in df_results.columns:
    cols_out.append('revenue_at_risk')

df_results[cols_out].to_csv("churn_predictions.csv", index=False)

print("\n✅ churn_predictions.csv created successfully!")
