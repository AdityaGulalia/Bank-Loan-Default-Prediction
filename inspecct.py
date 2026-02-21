import pandas as pd
pd.set_option('display.max_columns', None)  
df = pd.read_csv("Data/train.csv")

print("First 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique values:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")
    
print(df['Loan Status'].value_counts())