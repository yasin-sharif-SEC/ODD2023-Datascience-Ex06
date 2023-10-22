# Ex 6 Feature Transformation

# AIM
To read the given data and reduce the skewness to form a normal distribution.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.<br/>
There are three types of transformation.
* Function Tranformation
  * Log Tranformation
  * Reciprocal Transformation
  * Square root Transformation
* Power Tranformation
  * Box Cox Tansformation
  * Yeo-Johnson Transformation
* Quantile Transformation


# ALGORITHM
### Step 1
Read the given data.
### Step 2
Perform Q-Q plot to reveal the skewness in a feature.
### Step 3
Apply all possible feature transformation technique.
### Step 4
Determine the best technique.
### Step 5
Using that technique replace the transformed feature to the data

# CODE
```python
# import necessary modules and functions
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np
from sklearn.preprocessing import QuantileTransformer,PowerTransformer

# reading the file
df=pd.read_csv('Data_to_Transform.csv')
df

# highly positive skewed column
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# feature transformation for highly positive skewed column
# log transformation
df1=np.log(df['Highly Positive Skew'])
sm.qqplot(df1,fit=True,line='45')
plt.title('Log Transformation')
plt.show()

# reciprocal transformation
df2=1/df['Highly Positive Skew']
sm.qqplot(df2,fit=True,line='45')
plt.title('Reciprocal Transformation')
plt.show()

# square root transformation
df3=np.sqrt(df['Highly Positive Skew'])
sm.qqplot(df3,fit=True,line='45')
plt.title('Square Root Transformation')
plt.show()

# yeo-johnson transformation
tf=PowerTransformer("yeo-johnson")
df4=tf.fit_transform(df[['Highly Positive Skew']])
sm.qqplot(df4,fit=True,line='45')
plt.title('Yeo-Johnson Transformation')
plt.show()

# moderate positive skewed column
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

# feature transformation for moderate positive skewed column
# log transformation
df1=np.log(df['Moderate Positive Skew'])
sm.qqplot(df1,fit=True,line='45')
plt.title('Log Transformation')
plt.show()

# reciprocal transformation
df2=1/df['Moderate Positive Skew']
sm.qqplot(df2,fit=True,line='45')
plt.title('Reciprocal Transformation')
plt.show()

# square root transformation
df3=np.sqrt(df['Moderate Positive Skew'])
sm.qqplot(df3,fit=True,line='45')
plt.title('Square Root Transformation')
plt.show()

# yeo-johnson transformation
tf=PowerTransformer("yeo-johnson")
df4=tf.fit_transform(df[['Moderate Positive Skew']])
sm.qqplot(df4,fit=True,line='45')
plt.title('Yeo-Johnson Transformation')
plt.show()

# moderate negative skewed column
sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

# feature transformation for moderate negative skewed column
# yeo johnson tranformation
tf3=PowerTransformer('yeo-johnson')
df11=tf3.fit_transform(df[['Moderate Negative Skew']])
sm.qqplot(df11,line='45')
plt.title('Yeo-Johnson for Moderate Negative Skew Data')
plt.show()

# quantile transformation
qt1=QuantileTransformer(output_distribution='normal')
df12=qt1.fit_transform(df[['Moderate Negative Skew']])
sm.qqplot(df12,line='45')
plt.title('Quantile for Moderate Negative Skew Data')
plt.show()

# higly negative skewed column
sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

# feature transformation for higly negative skewed column
# yeo johnson tranformation
tf4=PowerTransformer('yeo-johnson')
df13=tf4.fit_transform(df[['Highly Negative Skew']])
sm.qqplot(df13,line='45')
plt.title('Yeo-Johnson for Highly Negative Skew Data')
plt.show()

# quantile transformation
qt2=QuantileTransformer(output_distribution='normal')
df14=qt2.fit_transform(df[['Highly Negative Skew']])
sm.qqplot(df14,line='45')
plt.title('Quantile for Highly Negative Skew Data')
plt.show()
```

# OUTPUT
### Highly positive skew column
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/9452e339-d31f-4e39-a3e4-5f5cc6a00d61)

### Feature transformation for highly positive skew column
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/5fa372d9-b355-4720-a233-a7d24286c9e2)
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/24b1599b-5f5e-473b-9d50-82ba3490ae63)
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/24004813-0645-4c7a-9534-1cd3ea8f0977)
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/0e223100-ceea-4364-a896-22399c39ded8)

### Moderate positive skew column
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/7449beb2-9903-4658-b760-4a56744c833c)

### Feature transformation for moderate positive skew column
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/43b28061-6621-486c-aaa6-deb1c4aa08a3)
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/dd73e278-8d7d-43c7-b6f5-e74d49d1759b)
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/05b7d5f9-b05a-4795-9f2a-003a0bdc22f5)
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/f2cecf2f-66b4-45a5-8054-a40dc26e1c2f)

### Moderate negative skew column
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/5354ba83-031a-4de9-991b-897d13741517)

### Feature transformation for moderate negative skew column
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/608634cb-7720-47eb-9e48-902eac3d2b28)
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/af32fae6-3d2e-4a7e-9450-1d3fd4302487)

### Highly negative skew column
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/8c8d0fa9-0520-4d88-89b9-24160c2b9e4e)

### Feature transformation for highly negative skew column
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/9f6e89b9-ce84-48e2-8536-76b6e3fc7581)
![download](https://github.com/yasin-sharif-SEC/ODD2023-Datascience-Ex06/assets/142985837/d5fb9534-8c6e-4f10-a707-30a34b9d2425)

# RESULT
Thus, the given data is read and the skewness is reduced.







