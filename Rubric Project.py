# Import Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Replacing 0 and 1 with "No" and "Yes" in Hypertension and Heart Disease columns
df2["hypertension"].replace([0,1], ["No","Yes"], inplace=True)
df2["heart_disease"].replace([0,1], ["No","Yes"], inplace=True)
df2["stroke"].replace([0,1], ["No","Yes"], inplace=True)

print(df2.head())

# Sort df2 by descending bmi
df2_bmi = df2.sort_values('bmi', ascending=False)
print(df2_bmi.head())

# Sort df2 by gender and descending age and bmi
df2_gen_age_bmi = df2.sort_values(['gender', 'age', 'bmi'], ascending=[True, False, False])
print(df2_gen_age_bmi.head())

# Sort df2 by gender and work type
df2_gen_work = df2.sort_values(['gender', 'work_type'], ascending=[False, False])
print(df2_gen_work.head())

# Index df2 by gender and sort
df2_srt = df2.set_index('gender').sort_index()
print(df2_srt.head())

# Use .loc() to subset df2_srt
print(df2_srt.loc["Male", ["age", "hypertension", "heart_disease", "bmi", "stroke"]])

# Categorising BMI into "Underweight", "Normal Weight", "Overweight" and "Obese"
Results = []

for i in df2["bmi"]:

    if (i < 18.5):
        Results.append("Underweight")

    elif (i >= 18.5) & (i <= 24.9):
        Results.append("Normal Weight")

    elif (i >= 25) & (i <= 29.9):
        Results.append("Overweight")

    elif (i >= 30):
        Results.append("Obese")

Results2 = pd.DataFrame(Results, columns=["bmi category"])

df2["bmi category"] = Results2
print(df2.head(15))
print(df2)

# Iterate over rows of df2
for lab, row in df2.iterrows():
    print(lab)
    print(row)

# Code for loop that adds pensioner column
for lab, row in df2.iterrows():
    df2.loc[lab, "pensioner"] = row['age'] >= 66

print(df2)

# Calculate the number of female and male observations to see if one outweighs the other
def somecalculation(x):
    return (df['gender'] == x).sum()

print(somecalculation('Male'))
print(somecalculation('Female'))

# As female observations outweigh males, create male and female dataframes with an equal
# number of observations and merge to form a new dataframe
male = df2[df2["gender"] == "Male"].head(2100)

female = df2[df2["gender"] == "Female"].head(2100)

df3 = pd.concat([male, female], axis=0, join='outer', ignore_index=True)

print(df3.head(10))

print(df3.shape)

# Converting age column in female data to list using Series.tolist()
female_age = female["age"].tolist()

print("Converting age to list:")

# displaying list
print(female_age)

# Converting stroke column in female data to list using Series.tolist()
female_stroke = female["stroke"].tolist()

print("Converting stroke to list:")

# displaying list
print(female_stroke)

# Convert female ages and stroke stats to numpy arrays: np_female_age, np_female_stroke
np_female_age = np.array(female_age)
np_female_stroke = np.array(female_stroke)

# Age of women who suffered a stroke
female_stroke_yes = np_female_age[np_female_stroke == "Yes"]

# Converting age column in male data to list using Series.tolist()
male_age = male["age"].tolist()

print("Converting age to list:")

# displaying list
print(male_age)

# Converting stroke column in male data to list using Series.tolist()
male_stroke = male["stroke"].tolist()

print("Converting stroke to list:")

# displaying list
print(male_stroke)

# Convert male ages and stroke stats to numpy arrays: np_male_age, np_male_stroke
np_male_age = np.array(male_age)
np_male_stroke = np.array(male_stroke)

# Age of men who suffered a stroke
male_stroke_yes = np_male_age[np_male_stroke == "Yes"]

# Print out the mean age of women who suffered a stroke
print(np.mean(female_stroke_yes))

# Print out the mean age of men who suffered a stroke
print(np.mean(male_stroke_yes))



