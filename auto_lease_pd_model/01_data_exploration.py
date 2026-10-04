# Import Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Display Setting of data
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Laod Data

df= pd.read_excel('Pakistan_Auto_Lease_PD_Dataset.xlsx', sheet_name='Loan_Data')
# Check Data
#print(df.head())
#print("\nshape of data:")
#print(df.shape)
#print("\nColumn Names:")
#print(df.columns.tolist())
#print("n\Data Type:")
#print(df.dtypes)
#print('\nMissing Values:')
#print(df.isnull().sum())


#print("\nDefault Distribution:")
#print(df['Default'].value_counts())
#print(f"\nDefault Rate: {df['Default'].mean()*100:.1f}%")

# chart
# Drwa Figure (blank)
plt.figure(figsize=(8,5))
default_by_type = df.groupby('Borrower_Type')['Default'].mean() * 100

 

bars = plt.bar(default_by_type.index,
               default_by_type.values,
               color =['#1F4E79','#2E75B6','#9DC3E6'])

plt.title('Default Rate by Borrower Type', fontsize=14, fontweight='bold')
plt.ylabel('Default Rate (%)')
plt.xlabel('Borrower Type')

for bar, val in zip(bars,default_by_type.values):   
    plt.text(
        bar.get_x() + bar.get_width()/2,  
        bar.get_height() + 0.3,
        f'{val:.1f}%',
        ha='center',
        fontweight= 'bold'
    )
plt.tight_layout()
plt.savefig('chart1_Defalut_by_Borrower_Type.png', dpi=150)
#


plt.figure(figsize=(8,5))
ecib_order = ['A','B','C','D','E',]


default_by_ecib = df.groupby('eCIB_Category')['Default'].mean().reindex(ecib_order) * 100
colors = ['#375623','#70AD47','#FFC000','#C00000','#7030A0']
bars = plt.bar(ecib_order,
              default_by_ecib.values,
              color=colors)  
plt.title('Default rate by ecib category',fontsize=14,fontweight='bold')
plt.xlabel ('ecib category')
plt.ylabel ('Default Rate (%)')
plt.ylim(0, 80)

for bar, val in zip(bars, default_by_ecib.values):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 2,
        f'{val:.1f}%',
        ha='center',
        fontweight='bold',
        color = 'black',
        fontsize=10
    )

plt.tight_layout()
plt.savefig('chart2_defalut_by_Ecib.png', dpi=150)
#   


plt.figure(figsize=(9,5))
default_by_DBR = df[df['Default']==1]['DBR_Percent']
nodefault_by_DBR = df[df['Default']==0]['DBR_Percent']

plt.hist(nodefault_by_DBR, bins=20, alpha=0.6, 
         color='#2E75B6', label='No Default')
plt.hist(default_by_DBR, bins=20, alpha=0.6, 
         color='#C00000', label='Default')

plt.axvline(x=40, color='black', linestyle='--', 
            linewidth=2, label='DBR Limit 40%')

plt.title('DBR Distribution - Default VS Non-Default', 
          fontsize=14, fontweight='bold')

plt.xlabel('DBR %')
plt.ylabel('Number of Borrowers')
plt.legend()

plt.tight_layout()
plt.savefig('chart3_dbr_distribution.png', dpi=150)
plt.show()

