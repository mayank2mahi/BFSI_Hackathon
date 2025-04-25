import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv("german_credit_data.csv", delim_whitespace=True, header=None)

columns = ["Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings",
           "EmploymentSince", "InstallmentRate", "PersonalStatusSex", "OtherDebtors",
           "ResidenceSince", "Property", "Age", "OtherInstallmentPlans", "Housing",
           "NumberCredits", "Job", "PeopleLiable", "Telephone", "ForeignWorker", "Risk"]
df.columns = columns

cat_cols = df.select_dtypes(include="object").columns.tolist()
df_encoded = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])

df_encoded["Risk"] = df_encoded["Risk"].map({1: 0, 2: 1})

X = df_encoded.drop("Risk", axis=1)
y = df_encoded["Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

with open("credit_risk_app/model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)
