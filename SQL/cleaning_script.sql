## PART 1 — Load Your CSV into MySQL Workbench (Mac Version)
## PART 2 — Verify the Data Loaded Correctly

## PART 3 
#(STEP 1) — Clean the Data in MySQL
DROP TABLE IF EXISTS customers_clean;

CREATE TABLE customers_clean AS
SELECT
  customerID,
  gender,

  /* SeniorCitizen cleaned safely */
  CASE
      WHEN LOWER(SeniorCitizen) = '1' THEN 1
      WHEN LOWER(SeniorCitizen) = 'yes' THEN 1
      WHEN LOWER(SeniorCitizen) = '0' THEN 0
      WHEN LOWER(SeniorCitizen) = 'no' THEN 0
      ELSE NULL
  END AS senior_citizen,

  /* Binary columns */
  CASE WHEN LOWER(Partner)='yes' THEN 1 WHEN LOWER(Partner)='no' THEN 0 ELSE NULL END AS partner,
  CASE WHEN LOWER(Dependents)='yes' THEN 1 WHEN LOWER(Dependents)='no' THEN 0 ELSE NULL END AS dependents,

  /* Tenure safe conversion */
  CASE
      WHEN tenure = '' THEN NULL
      ELSE CAST(tenure AS SIGNED)
  END AS tenure_months,

  /* PhoneService */
  CASE WHEN LOWER(PhoneService)='yes' THEN 1 WHEN LOWER(PhoneService)='no' THEN 0 ELSE NULL END AS phone_service,

  /* Keep categorical columns as text */
  MultipleLines,
  InternetService,
  OnlineSecurity,
  OnlineBackup,
  DeviceProtection,
  TechSupport,
  StreamingTV,
  StreamingMovies,
  Contract,

  /* PaperlessBilling */
  CASE WHEN LOWER(PaperlessBilling)='yes' THEN 1 WHEN LOWER(PaperlessBilling)='no' THEN 0 ELSE NULL END AS paperless_billing,

  PaymentMethod,

  /* Charges — converted safely */
  CASE WHEN MonthlyCharges = '' THEN NULL ELSE CAST(MonthlyCharges AS DECIMAL(10,2)) END AS monthly_charges,
  CASE WHEN TotalCharges = '' THEN NULL ELSE CAST(TotalCharges AS DECIMAL(12,2)) END AS total_charges,

  /* Churn */
  CASE WHEN LOWER(Churn)='yes' THEN 1 WHEN LOWER(Churn)='no' THEN 0 ELSE NULL END AS churn

FROM customers_raw;









UPDATE customers_clean
SET total_charges = monthly_charges * tenure_months
WHERE total_charges IS NULL AND monthly_charges IS NOT NULL AND tenure_months IS NOT NULL;



## PART 3 
#(STEP 2) — Fix customers with missing TotalCharges

UPDATE customers_clean
SET total_charges = monthly_charges * tenure_months
WHERE total_charges IS NULL
  AND customerID IS NOT NULL;
  
  
## PART 3 
#(STEP 3- IT HAS 4 STEPS)  — Add derived features
#1- ADDING NEW COLUMNS 
ALTER TABLE customers_clean
ADD COLUMN has_multiple_services INT,
ADD COLUMN is_long_term_contract INT,
ADD COLUMN tenure_group VARCHAR(32);

#2- Fill has_multiple_services
UPDATE customers_clean
SET has_multiple_services =
  (CASE WHEN OnlineSecurity='Yes' THEN 1 ELSE 0 END) +
  (CASE WHEN OnlineBackup='Yes' THEN 1 ELSE 0 END) +
  (CASE WHEN DeviceProtection='Yes' THEN 1 ELSE 0 END) +
  (CASE WHEN TechSupport='Yes' THEN 1 ELSE 0 END) +
  (CASE WHEN StreamingTV='Yes' THEN 1 ELSE 0 END) +
  (CASE WHEN StreamingMovies='Yes' THEN 1 ELSE 0 END);


#3 — Fill is_long_term_contract
UPDATE customers_clean
SET is_long_term_contract =
    CASE WHEN Contract IN ('One year','Two year') THEN 1 ELSE 0 END;

#4 — Fill tenure_group
UPDATE customers_clean
SET tenure_group = CASE
    WHEN tenure_months <= 1 THEN '0-1'
    WHEN tenure_months <= 6 THEN '1-6'
    WHEN tenure_months <= 12 THEN '6-12'
    WHEN tenure_months <= 24 THEN '12-24'
    WHEN tenure_months <= 48 THEN '24-48'
    ELSE '48+'
END;



SELECT tenure_group, is_long_term_contract, has_multiple_services
FROM customers_clean
LIMIT 10;


## PART3 
#(STEP4)- KPI SQL queries (run in MySQL Workbench)
# 1. Create a small KPI table (run this SQL):
CREATE TABLE kpis AS
SELECT
  COUNT(*) AS total_customers,
  SUM(CASE WHEN churn=1 THEN 1 ELSE 0 END) AS churned_customers,
  ROUND(100 * SUM(CASE WHEN churn=1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_percent,
  ROUND(AVG(monthly_charges),2) AS avg_monthly_charges,
  ROUND(AVG(tenure_months),2) AS avg_tenure_months
FROM customers_clean;


SELECT * FROM kpis;

# 2. Quick checks (run these SELECTs):
SELECT * FROM kpis;
SELECT Contract, COUNT(*) AS cnt, ROUND(100*SUM(churn)/COUNT(*),2) AS churn_pct
FROM customers_clean
GROUP BY Contract
ORDER BY churn_pct DESC;
SELECT PaymentMethod, COUNT(*) AS cnt, ROUND(100*SUM(churn)/COUNT(*),2) AS churn_pct
FROM customers_clean
GROUP BY PaymentMethod
ORDER BY churn_pct DESC;



## PART 4 — After SQL cleaning → Load into Python
import pandas as pd
import mysql.connector

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yourpassword",
    database="churn_db"
)

query = "SELECT * FROM customers_clean;"
df = pd.read_sql(query, connection)

