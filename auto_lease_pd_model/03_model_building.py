import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_excel('Pakistan_Auto_Lease_PD_Dataset.xlsx', 
                   sheet_name='Loan_Data')

df = df.drop('Borrower_ID', axis=1)

df['Guarantor'] = df['Guarantor'].map({'Yes': 1, 'No': 0})
df['Previous_Relationship'] = df['Previous_Relationship'].map({'Yes': 1, 'No': 0})

# E=4 (unknown/medium risk), D=5 (highest risk — confirmed default)
ecib_map = {'A': 1, 'B': 2, 'C': 3, 'E': 4, 'D': 5}
df['eCIB_Category'] = df['eCIB_Category'].map(ecib_map)

df = pd.get_dummies(df, columns=['Borrower_Type', 'Product_Type'], drop_first=True)
df = df.drop('LTV_Percent', axis=1)

X = df.drop('Default', axis=1)
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000, random_state=42, solver='lbfgs', class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nModel Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.3f}")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

feature_names = X.columns.tolist()
coefficients  = model.coef_[0]

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.barh(feature_importance['Feature'],
         feature_importance['Abs_Coefficient'],
         color='#1F4E79')
plt.title('Feature Importance — Logistic Regression', fontsize=14, fontweight='bold')
plt.xlabel('Coefficient Absolute Value')
plt.tight_layout()
plt.savefig('chart4_feature_importance.png', dpi=150)
plt.show()

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("\nXGBoost Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb)*100:.1f}%")
print(f"AUC Score: {roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1]):.3f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred_xgb))