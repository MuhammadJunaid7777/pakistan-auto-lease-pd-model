 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_excel('Pakistan_Auto_Lease_PD_Dataset.xlsx', sheet_name='Loan_Data')
#print("Original shape:", df.shape)

df= df.drop('Borrower_ID', axis=1)
#print("after dropping ID:", df.shape)

df['Guarantor'] = df['Guarantor'].map({'Yes': 1, 'No': 0})
df['Previous_Relationship'] = df['Previous_Relationship'].map({'Yes': 1, 'No': 0})

#print("\nGuarantor values:", df['Guarantor'].unique())
#print("Previous_Relationship values:", df['Previous_Relationship'].unique())

ecib_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
df['eCIB_Category'] = df['eCIB_Category'].map(ecib_map)
#print("\neCIB values:", df['eCIB_Category'].unique())

df = pd.get_dummies(df, columns=['Borrower_Type', 'Product_Type'], drop_first=True)
#print("\nNew shape:", df.shape)
#print("\nNew columns:", df.columns.tolist())

df = df.drop('LTV_Percent', axis=1)
#print("\nAfter dropping LTV:", df.shape)




X = df.drop('Default', axis=1)
y = df['Default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)