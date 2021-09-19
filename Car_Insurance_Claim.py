# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 11:15:01 2021

@author: Sherin Joseph
"""

import os

os.getcwd()  # 'C:\\Users\\Sherin Joseph'

os.chdir("C:\D Drive\Data Science\8. Python\Directory")

os.getcwd()  # 'C:\\D Drive\\Data Science\\8. Python\\Directory'

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


plt.interactive(False) # Interactive mode is turned OFF


# **************** Reading Data Set *******************

insurance_root = pd.read_csv("Car_Insurance_Claim.csv",na_values='')

insurance_root.shape  # (10000, 19) : 10000 rows and 19 columns

insurance_root.info()
'''
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 19 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   ID                   10000 non-null  int64  
 1   AGE                  10000 non-null  object 
 2   GENDER               10000 non-null  object 
 3   RACE                 10000 non-null  object 
 4   DRIVING_EXPERIENCE   10000 non-null  object 
 5   EDUCATION            10000 non-null  object 
 6   INCOME               10000 non-null  object 
 7   CREDIT_SCORE         9018 non-null   float64
 8   VEHICLE_OWNERSHIP    10000 non-null  float64
 9   VEHICLE_YEAR         10000 non-null  object 
 10  MARRIED              10000 non-null  float64
 11  CHILDREN             10000 non-null  float64
 12  POSTAL_CODE          10000 non-null  int64  
 13  ANNUAL_MILEAGE       9043 non-null   float64
 14  VEHICLE_TYPE         10000 non-null  object 
 15  SPEEDING_VIOLATIONS  10000 non-null  int64  
 16  DUIS                 10000 non-null  int64  
 17  PAST_ACCIDENTS       10000 non-null  int64  
 18  OUTCOME              10000 non-null  float64
 '''
 insurance_root.duplicated().sum()   # There are no duplicate values
 
 len(insurance_root['ID'].unique())  # 10000 --> Means the ID is unique and can be made as index column
 
 # New Data frame using the column 'ID' as index
 insurance = pd.read_csv("Car_Insurance_Claim.csv",na_values='',index_col=[0])
 
 
 # ************************************* Business problems *************************************
 
 # 1. Find the age distribution of the clients
 
 insurance = pd.read_csv("Car_Insurance_Claim.csv",na_values='',index_col=[0])
 
 age_categories = insurance['AGE'].value_counts()
 '''
 26-39    3063
 40-64    2931
 16-25    2016
 65+      1990
 '''
  
 
values = list(age_categories.values)
colors = ['b', 'g', 'r', 'y']
lbls = list(age_categories.index)
exp = (0.1, 0, 0, 0)

plt.pie(values, colors = colors, labels = values, autopct='%1.2f%%',explode = exp, counterclock=False, shadow=True)

plt.title('Age Group Distribution')

plt.legend(labels = lbls, loc=3,bbox_to_anchor=(1,0))

plt.show()


# 1.1. Find the DRIVING_EXPERIENCE distribution of the clients

driving_exp_categories = insurance['DRIVING_EXPERIENCE'].value_counts()

'''
0-9y      3530
10-19y    3299
20-29y    2119
30y+      1052
'''

categories=list(driving_exp_categories.index)
pos = np.arange(len(categories))  # 0 - 3
values = list(driving_exp_categories.values)

fig, ax = plt.subplots()
ax.bar(pos,values,color = ['c','m','brown','orange'],edgecolor='black')
for index,data in enumerate(values):
    plt.text(x=index , y =data+2 , s=f"{data}" , fontdict=dict(fontsize=14))
plt.tight_layout()

plt.xticks(pos, categories)
plt.xlabel('Experience', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Driving Experience of Clients',fontsize=18)

plt.show()


# 2. Find the INCOME type distribution of the clients

income_categories = insurance['INCOME'].value_counts()

'''
upper class      4336
middle class     2138
poverty          1814
working class    1712
'''
categories=list(income_categories.index)
pos = np.arange(len(categories))  # 0 - 3
values = list(income_categories.values)

fig, ax = plt.subplots()
ax.barh(pos,values,color = ['palegreen','pink','grey','mediumorchid'],edgecolor='black')
for index,data in enumerate(values):
    plt.text(x =data+1 ,y=index , s=f"{data}" , fontdict=dict(fontsize=14))
plt.tight_layout()

plt.yticks(pos, categories)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Income Category', fontsize=14)
plt.title('Income Category Distribution',fontsize=18)

plt.show()

# 2.1. Find the Postal code distribution of the clients

#insurance = pd.read_csv("Car_Insurance_Claim.csv",na_values='',index_col=[0])
 
 postalcodes = insurance['POSTAL_CODE'].value_counts()
 '''
 10238    6940
 32765    2456
 92101     484
 21217     120
 '''
  
 
values = list(postalcodes.values)
colors = ['teal', 'Chocolate', 'deepskyblue', 'fuchsia']
lbls = list(postalcodes.index)
exp = (0.1, 0, 0, 0)

plt.pie(values, colors = colors, labels = values, autopct='%1.2f%%',explode = exp, counterclock=False, shadow=False)

plt.title('Postal Code Distribution')

plt.legend(labels = lbls, loc=3,bbox_to_anchor=(1,0))

plt.show()


# 3. Find the CREDIT_SCORE distribution of the clients

 # Fixing CREDIT_SCORE by eliminating outliers and null values and converting to int type
 
 insurance["CREDIT_SCORE"] = insurance["CREDIT_SCORE"] * 1000 # changing decimal points
 
 insurance = insurance[(insurance.CREDIT_SCORE >= 200) & (insurance.CREDIT_SCORE <= 900)]
 
 insurance['CREDIT_SCORE'] = insurance['CREDIT_SCORE'].astype(int) # Changing data type to int
 
 insurance.shape  # (8917, 18)  ---> by eliminating outliers and na values in credit score

values = list(insurance['CREDIT_SCORE'].values)

plt.hist(values,35,histtype='bar',alpha=0.8, align='mid', color='g', label='Credit Score',ec='black')

plt.legend(loc=2)
plt.title('Histogram of Credit Score')
plt.xlabel('Credit Score', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()


# 4. Find the ANNUAL_MILEAGE distribution of the clients
 
 # Fixing missing values is ANNUAL_MILEAGE with average value of non-missing values
 
 insurance.ANNUAL_MILEAGE.isnull().sum() # 957 --> values with are missing
 
 insurance.loc[insurance.ANNUAL_MILEAGE.isnull(),['ANNUAL_MILEAGE']] = round(insurance[insurance.ANNUAL_MILEAGE.notnull()]['ANNUAL_MILEAGE'].mean())
 
 insurance.ANNUAL_MILEAGE.isnull().sum()  # No more missing values again
 
 
 accident_categories = insurance['ANNUAL_MILEAGE'].value_counts()
 
 
 box_plot_data = list(insurance['ANNUAL_MILEAGE']) # --> Nested List

 bp = plt.boxplot(box_plot_data,labels=['Annual Mileage'],patch_artist=True)
 ax = plt.axes()
 ax.set(facecolor = "slategrey")
 plt.title('Box Plot of Annual Mileage Distribution')
 plt.show()
 

 
# 5. Find the Average Credit Score according to the age group of the clients

 #insurance = pd.read_csv("Car_Insurance_Claim.csv",na_values='',index_col=[0])

# Fixing CREDIT_SCORE by eliminating outliers and null values and converting to int type
 
 insurance["CREDIT_SCORE"] = insurance["CREDIT_SCORE"] * 1000 # changing decimal points
 
 insurance = insurance[(insurance.CREDIT_SCORE >= 200) & (insurance.CREDIT_SCORE <= 900)]
 
 insurance['CREDIT_SCORE'] = insurance['CREDIT_SCORE'].astype(int) # Changing data type to int
 
 insurance.shape  # (8917, 18)  ---> by eliminating outliers and na values in credit score
 
 insurance.groupby('AGE')['CREDIT_SCORE'].mean().plot(kind = 'barh',color = ['darkslategray'])

 ax = plt.axes()
 ax.set(facecolor = "lightsalmon")
 plt.ylabel('Age Group', fontsize=14)
 plt.xlabel('Credit Score', fontsize=14)
 plt.title('Age group vs Credit Score',fontsize=18)
 plt.show()
 
 
# 6. Which class (INCOME) of clients made maximum accidents in the past

#insurance = pd.read_csv("Car_Insurance_Claim.csv",na_values='',index_col=[0])

income_accidents = insurance.groupby('INCOME')['PAST_ACCIDENTS'].sum()
'''
middle class     2195
poverty           550
upper class      6662
working class    1156
'''
categories=list(income_accidents.index)
pos = np.arange(len(categories))  # 0 - 3
values = list(income_accidents.values)

fig, ax = plt.subplots()
ax.barh(pos,values,color = ['darkorchid','olive','orangered','aqua'],edgecolor='black')
for index,data in enumerate(values):
    plt.text(x =data+1 ,y=index , s=f"{data}" , fontdict=dict(fontsize=14))
plt.tight_layout()

plt.yticks(pos, categories)
plt.xlabel('No. of Accidents in past', fontsize=14)
plt.ylabel('Income Category', fontsize=14)
plt.title('Income Class vs Total Accidents',fontsize=18)

plt.show()

 
 
# 7. Is there any relation between Annual Mileage and Speeding Violations

 #insurance = pd.read_csv("Car_Insurance_Claim.csv",na_values='',index_col=[0])

 insurance.ANNUAL_MILEAGE.isnull().sum() # 957 --> values with are missing
 
 insurance.loc[insurance.ANNUAL_MILEAGE.isnull(),['ANNUAL_MILEAGE']] = round(insurance[insurance.ANNUAL_MILEAGE.notnull()]['ANNUAL_MILEAGE'].mean())
 
 insurance.ANNUAL_MILEAGE.isnull().sum()  # No more missing values again

mileage = list(insurance['ANNUAL_MILEAGE'].values)
speed_viloation = list(insurance['SPEEDING_VIOLATIONS'].values)

fig = plt.figure()
fig.patch.set_facecolor('indianred')
plt.scatter(mileage,speed_viloation, c='b', marker='*') # marker='o' denotes the shape od points, "c" --> Colour
plt.xlabel('Annual Mileage', fontsize=16)
plt.ylabel('Speeding Violation', fontsize=16)
ax = plt.axes()
ax.set_facecolor("gold")
plt.title('Scatter plot - Annual Mileage vs Speeding Violation', fontsize=16)
plt.show()


# 8. Who made more claims :  male or female and based on their driving experience

#insurance = pd.read_csv("Car_Insurance_Claim.csv",na_values='',index_col=[0])

pd.crosstab(insurance.DRIVING_EXPERIENCE, insurance.GENDER).plot(kind = 'bar')
ax = plt.axes()
ax.set_facecolor("darkseagreen")

plt.xlabel('Driving Experience', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Driving Experience vs Gender',fontsize=20)

plt.show()

# 9. Which vehicle type had more claims across the different postal codes.
#insurance = pd.read_csv("Car_Insurance_Claim.csv",na_values='',index_col=[0])

pd.crosstab(insurance.POSTAL_CODE, insurance.VEHICLE_TYPE).plot(kind = 'barh',color=['darkorchid','blue']) #horizondal
ax = plt.axes()
ax.set_facecolor("slategrey")

plt.ylabel('Postal Code', fontsize=16)
plt.xlabel('Count', fontsize=16)
plt.title('Postal Code vs Vehicle Type',fontsize=20)
plt.show()


# 10. Which cars (based on vehicle year) had atleast one history of accidents
 insurance = insurance[insurance.PAST_ACCIDENTS >0]
 
 
 pd.crosstab(insurance.PAST_ACCIDENTS, insurance.VEHICLE_YEAR).plot(kind = 'bar',color=['darkorchid','blue']) #horizondal
ax = plt.axes()
ax.set_facecolor("slategrey")

plt.xlabel('History of Accidents', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Accident History vs Vehicle Year',fontsize=20)
plt.show()
 
 #11. Claims approval based on Age, Gender, Income, vehicle year,vehicle type and postal code
 
 #insurance = pd.read_csv("Car_Insurance_Claim.csv",na_values='',index_col=[0])
 insurance['OUTCOME'] = insurance["OUTCOME"].replace(0,"Not Approved")   
 insurance['OUTCOME'] = insurance["OUTCOME"].replace(1,"Approved")
 
 insurance['OUTCOME'].value_counts()
 
 
 pd.crosstab(insurance.AGE, insurance.OUTCOME).plot(kind = 'barh',color=['forestgreen','firebrick'])
 ax = plt.axes()
 ax.set_facecolor("dimgrey")
 plt.title('Age vs Approval',fontsize=20)
 plt.show()
 
 pd.crosstab(insurance.GENDER, insurance.OUTCOME).plot(kind = 'barh',color=['forestgreen','firebrick'])
 ax = plt.axes()
 ax.set_facecolor("dimgrey")
 plt.title('Gender vs Approval',fontsize=20)
 plt.show()
 
 pd.crosstab(insurance.INCOME, insurance.OUTCOME).plot(kind = 'barh',color=['forestgreen','firebrick'])
 ax = plt.axes()
 ax.set_facecolor("dimgrey")
 plt.title('Income Class vs Approval',fontsize=20)
 plt.show()
 
 pd.crosstab(insurance.VEHICLE_YEAR, insurance.OUTCOME).plot(kind = 'barh',color=['forestgreen','firebrick'])
 ax = plt.axes()
 ax.set_facecolor("dimgrey")
 plt.title('Vehicle Year vs Approval',fontsize=20)
 plt.show()

 pd.crosstab(insurance.VEHICLE_TYPE, insurance.OUTCOME).plot(kind = 'barh',color=['forestgreen','firebrick'])
 ax = plt.axes()
 ax.set_facecolor("dimgrey")
 plt.title('Vehicle Type vs Approval',fontsize=20)
 plt.show()
 
 pd.crosstab(insurance.POSTAL_CODE, insurance.OUTCOME).plot(kind = 'barh',color=['forestgreen','firebrick'])
 ax = plt.axes()
 ax.set_facecolor("dimgrey")
 plt.title('Postal Code vs Approval',fontsize=20)
 plt.show()
 
 