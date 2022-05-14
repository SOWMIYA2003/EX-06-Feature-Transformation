# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM
### STEP 1
Read the given Data.
### STEP 2
Clean the Data Set using Data Cleaning Process.
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set.
### STEP 4
Save the data to the file.

## Types Of Transformations For Better Normal Distribution:

### Log Transformation

Log transformations make our data close to a normal distribution but are not able to exactly abide by a normal distribution.This transformation is mostly applied to right-skewed data.

### Square-Root Transformation

Square-Root transformation will give a moderate effect on distribution. The main advantage of square root transformation is, it can be applied to zero values.

### Reciprocal Transformation

Reciprocal transformation reverses the order among values of the same sign, so large values become smaller and vice-versa.This transformation is not defined for zero.It is a powerful transformation with a radical effect.

### Box-Cox Transformation

Box-cox requires the input data to be strictly positive(not even zero is acceptable).All the values of lambda vary from -5 to 5 are considered and the best value for the data is selected. The “Best” value is one that results in the best skewness of the distribution.Box-cox function reduced the skewness and it is almost equal to zero.

### Yeo-Johnson Transformation

Yeo-Johnson is best suited for features that have zeroes or negative values
YEO-JOHNSON TRANSFORMATION:  It is a variation of the Box-Cox transform.

# DataSet 1- Data_To_Transfer.csv
# CODE
```
Developed By: Sowmiya N.
Register Number : 212221230106.
```

```
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt

df=pd.read_csv("Data_to_Transform.csv")
df

df1=df.copy()
log_ModPositive = np.log1p(df1["Moderate Positive Skew"])
log_HighPositive = np.log1p(df1["Highly Positive Skew"])
log_ModNegative = np.log1p(df1["Moderate Negative Skew"])
log_HighNegative = np.log1p(df1["Highly Negative Skew"])
df1.insert(1,"log_ModPositive",log_ModPositive)
df1.insert(3,"log_HighPositive",log_HighPositive)
df1.insert(5,"log_ModNegative",log_ModNegative)
df1.insert(7,"log_HighNegative",log_HighNegative)
df1

df1["Moderate Positive Skew"].plot(kind = 'hist')
df1["log_ModPositive"].plot(kind = 'hist')
df1["Highly Positive Skew"].plot(kind = 'hist')
df1["log_HighPositive"].plot(kind = 'hist')
df1["Moderate Negative Skew"].plot(kind = 'hist')
df1["log_ModNegative"].plot(kind = 'hist')
df1["Highly Negative Skew"].plot(kind = 'hist')
df1["log_HighNegative"].plot(kind = 'hist')

df2=df.copy()
sqrt_ModPositive = df["Moderate Positive Skew"]**(1/2)
sqrt_HighPositive = df["Highly Positive Skew"]**(1/2)
sqrt_ModNegative = df["Moderate Negative Skew"]**(1/2)
sqrt_HighNegative = df["Highly Negative Skew"]**(1/2)
df2.insert(1,"sqrt_ModPositive",sqrt_ModPositive)
df2.insert(3,"sqrt_HighPositive",sqrt_HighPositive)
df2.insert(5,"sqrt_ModNegative",sqrt_ModNegative)
df2.insert(7,"sqrt_HighNegative",sqrt_HighNegative)
df2

df2["Moderate Positive Skew"].plot(kind = 'hist')
df2["sqrt_ModPositive"].plot(kind = 'hist')
df2["Highly Positive Skew"].plot(kind = 'hist')
df2["sqrt_HighPositive"].plot(kind = 'hist')
df2["Moderate Negative Skew"].plot(kind = 'hist')
df2["sqrt_ModNegative"].plot(kind = 'hist')
df2["Highly Negative Skew"].plot(kind = 'hist')
df2["sqrt_HighNegative"].plot(kind = 'hist')

df3=df.copy()
reciprocal_ModPositive = 1/df["Moderate Positive Skew"]
reciprocal_HighPositive = 1/df["Highly Positive Skew"]
reciprocal_ModNegative = 1/df["Moderate Negative Skew"]
reciprocal_HighNegative = 1/df["Highly Negative Skew"]
df3.insert(1,"reciprocal_ModPositive",reciprocal_ModPositive)
df3.insert(3,"reciprocal_HighPositive",reciprocal_HighPositive)
df3.insert(5,"reciprocal_ModNegative",reciprocal_ModNegative)
df3.insert(7,"reciprocal_HighNegative",reciprocal_HighNegative)
df3

df3["reciprocal_ModPositive"].plot(kind = 'hist')
df3["reciprocal_HighPositive"].plot(kind = 'hist')
df3["reciprocal_ModNegative"].plot(kind = 'hist')
df3["reciprocal_HighNegative"].plot(kind = 'hist')

from scipy.stats import boxcox
df4=df.copy()
bcx_ModPositive, lam = boxcox(df["Moderate Positive Skew"])
bcx_HighPositive, lam = boxcox(df["Highly Positive Skew"])
df4.insert(1,"bcx_ModPositive",bcx_ModPositive)
df4.insert(3,"bcx_HighPositive",bcx_HighPositive)
df4

df4["bcx_ModPositive"].plot(kind = 'hist')
df4["bcx_HighPositive"].plot(kind = 'hist')

from scipy.stats import yeojohnson
df5=df.copy()
yf_ModPositive, lam = yeojohnson(df["Moderate Positive Skew"])
yf_HighPositive, lam = yeojohnson(df["Highly Negative Skew"])
yf_ModNegative, lam = yeojohnson(df["Moderate Negative Skew"])
yf_HighNegative, lam = yeojohnson(df["Highly Negative Skew"])
df5.insert(1,"yf_ModPositive",yf_ModPositive)
df5.insert(3,"yf_HighPositive",yf_HighPositive)
df5.insert(5,"yf_ModNegative",yf_ModNegative)
df5.insert(7,"yf_HighNegative",yf_HighNegative)
df5

df5["yf_ModPositive"].plot(kind = 'hist')
df5["yf_HighPositive"].plot(kind = 'hist')
df5["yf_ModNegative"].plot(kind = 'hist')
df5["yf_HighNegative"].plot(kind = 'hist')

```
# OUPUT

### Initial DataFrame:
![op](./a1.png)

### Applying Log Transformation Method to the given DataFrame:
![op](./a2.png)

### Column - Moderate Positive Skew

### Original:
![op](./a4.png)
### Log Transformed:
![op](./a5.png)

### Column - Highly Positive Skew

### Original:
![op](./a6.png)
### Log Transformed:
![op](./a7.png)

### Column - Moderate Negative Skew

### Original:
![op](./a8.png)
### Log Transformed:
![op](./a9.png)

### Column - Highly Negative Skew

### Original:
![op](./a10.png)
### Log Transformed:
![op](./a11.png)

### Applying Square-Root Transformation Method to the given DataFrame:
![op](./z1.png)

### Column - Moderate Positive Skew

### Original:
![op](./a4.png)
### Square-Root Transformed:
![op](./z3.png)

### Column - Highly Positive Skew

### Original:
![op](./a6.png)
### Square-Root Transformed:
![op](./z4.png)

### Column - Moderate Negative Skew

### Original:
![op](./a8.png)
### Square-Root Transformed:
![op](./z5.png)

### Column - Highly Negative Skew

### Original:
![op](./a10.png)
### Square-Root Transformed:
![op](./z6.png)


### Applying Reciprocal Transformation Method to the given DataFrame:
![op](./s1.png)

### Column - Moderate Positive Skew

### Original:
![op](./a4.png)
### Inverse Transformed:
![op](./s3.png)

### Column - Highly Positive Skew

### Original:
![op](./a6.png)
### Inverse Transformed:
![op](./s4.png)

### Column - Moderate Negative Skew

### Original:
![op](./a8.png)
### Inverse Transformed:
![op](./s5.png)

### Column - Highly Negative Skew

### Original:
![op](./a10.png)
### Inverse Transformed:
![op](./s6.png)

### Applying Box-Cox Transformation Method to the given DataFrame:
```
Box-cox requires the input data to be STRICTLY POSITIVE (not even zero is acceptable).
```
![op](./box.png)
### Column - Moderate Positive Skew

### Original:
![op](./a4.png)
### Box-Cox Transformed:
![op](./h2.png)

### Column - Highly Positive Skew

### Original:
![op](./a6.png)
### Box-Cox Transformed:
![op](./h3.png)

### Applying Yeo-Johnson Transformation Method to the given DataFrame:
![op](./b1.png)
### Column - Moderate Positive Skew

### Original:
![op](./a4.png)
### Y-J Transformed:
![op](./b5.png)

### Column - Highly Positive Skew

### Original:
![op](./a6.png)
### Y-J Transformed:
![op](./b4.png)

### Column - Moderate Negative Skew

### Original:
![op](./a8.png)
### Y-J Transformed:
![op](./b2.png)

### Column - Highly Negative Skew

### Original:
![op](./a10.png)
### Y-J Transformed:
![op](./b3.png)

## Result
 Various feature transformation techniques are performed on a given dataset for the better fit of normality successfully.


