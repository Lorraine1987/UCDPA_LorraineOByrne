# Import Modules
import pandas as pd
import numpy as np

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

print(df.head())

print(df.info())

print(df.shape)

# Check for missing values
missing_values = df.isnull()
print(missing_values)

print(missing_values.sum())

# Replace missing values with average
df2 = df.fillna(df.mean())
print(df2.head())
print(df2.shape)



