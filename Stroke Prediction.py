# Import Modules
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Import API: number of people in space
data = requests.get("http://api.open-notify.org/astros.json")
data = data.json()

print(data)

for p in data['people']:
    print(p['name'])

# Commence EDA of Stroke Prediction Dataset
# Import csv file: 'healthcare-dataset-stroke-data.csv'
def importdata(filename):
    return pd.read_csv(filename)

df = importdata('healthcare-dataset-stroke-data.csv')

print(df.head())

print(df.info())

print(df.shape)

# Check for missing values
def missing_values(data):
    return df.isnull()

print(missing_values(df))

print(missing_values(df).sum())

# Replace missing values with average
def replace_avg(data):
    return data.fillna(data.mean())

df2 = replace_avg(df)

print(df2.head())
print(df2.shape)

# Replacing 0 and 1 with "No" and "Yes" in Hypertension and Heart Disease columns
df2["hypertension"].replace([0, 1], ["No", "Yes"], inplace=True)
df2["heart_disease"].replace([0, 1], ["No", "Yes"], inplace=True)

print(df2.head())

# Remove "other" observation from gender
df2.drop(df2[df2['gender'] == 'Other'].index, inplace=True)
df2['gender'].unique()

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

female_age = female["age"].tolist()
# Converting age column in female data to list using Series.tolist()

print("Converting age to list:")

# displaying list
print(female_age)

# Converting stroke column in female data to list using Series.tolist()
female_stroke = female["stroke"].tolist()

print("Converting stroke to list:")

# displaying list
print(female_stroke)

# display average age of women
print(np.mean(female_age))

# Convert female ages and stroke stats to numpy arrays: np_female_age, np_female_stroke
np_female_age = np.array(female_age)
np_female_stroke = np.array(female_stroke)

# Age of women who suffered a stroke
female_stroke_yes = np_female_age[np_female_stroke == 1]

# Converting age column in male data to list using Series.tolist()
male_age = male["age"].tolist()

print("Converting age to list:")

# displaying list
print(male_age)

# display average age of men
print(np.mean(male_age))

# Converting stroke column in male data to list using Series.tolist()
male_stroke = male["stroke"].tolist()

print("Converting stroke to list:")

# displaying list
print(male_stroke)

# Convert male ages and stroke stats to numpy arrays: np_male_age, np_male_stroke
np_male_age = np.array(male_age)
np_male_stroke = np.array(male_stroke)

# Age of men who suffered a stroke
male_stroke_yes = np_male_age[np_male_stroke == 1]

# Print out the mean age of women who suffered a stroke
print(np.mean(female_stroke_yes))

# Print out the mean age of men who suffered a stroke
print(np.mean(male_stroke_yes))

# Pivot for mean and median bmi for stroke occurence
mean_med_bmi_stroke = df3.pivot_table(values='bmi', index='stroke', aggfunc=[np.mean, np.median])
print(mean_med_bmi_stroke)

# Pivot for mean age for stroke occurence
mean_age_stroke = df3.pivot_table(values='age', index='stroke')
print(mean_age_stroke)

# Pivot for mean age for stroke occurence factoring hypertension
mean_age_stroke_hyp = df3.pivot_table(values='age', index='stroke', columns='hypertension')
print(mean_age_stroke_hyp)

# Pivot for mean age for stroke occurence factoring smoking status
mean_age_stroke_smoke = df3.pivot_table(values='age', index='stroke', columns='smoking_status')
print(mean_age_stroke_smoke)

# Pivot for mean age for stroke occurence factoring work type
mean_age_stroke_work = df3.pivot_table(values='age', index='stroke', columns='work_type', fill_value=0)
print(mean_age_stroke_work)

# Visualise Occurence of Stroke by Gender
g = sns.countplot(x='gender', data=df3, hue='stroke')
g.set_title("Probability of Stroke by Gender")
plt.show()

# Visualise Average Age of Stroke Patients by Gender
g1 = sns.barplot(x='gender', y='age', data=df3, hue='stroke')
g1.set_title("Average Age of Stroke Patients by Gender")
plt.show()

# Visualise Distribution of Age
g2 = sns.FacetGrid(df3, col="stroke", row="gender")
g2.map_dataframe(sns.histplot, x="age", binwidth=10)
g2.set_axis_labels("Age", "Count")
plt.show()

# Visualise Average Bmi of Stroke Patients by Gender
g3 = sns.barplot(x='gender', y='bmi', data=df3, hue='stroke')
g3.set_title("Average Bmi of Stroke Patients by Gender")
plt.show()

# Is there a relationship between age and bmi
g4 = sns.relplot(x='age', y='bmi', data=df3, kind='scatter',
            hue='gender', col='stroke')
g4.fig.subplots_adjust(top=0.9, bottom=0.1)
g4.fig.suptitle("Relationship between Age and Bmi when predicting Stroke")
plt.show()

# Visualise Occurence of Strokes per Smoking Status
g5 = sns.countplot(x='stroke', data=df3, hue='smoking_status')
g5.set_title("Occurence of Stroke per Smoking Status")
plt.show()

# Visualise Occurence of Strokes with or without Heart Disease
g6 = sns.countplot(x='stroke', data=df3, hue='heart_disease')
g6.set_title("Occurence of Stroke w/wo Heart Disease ")
plt.show()

# Visualise Occurence of Strokes per Bmi Category
g7 = sns.countplot(x='stroke', data=df3, hue='bmi category')
plt.show()








