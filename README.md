## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION
  
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:

Developed by :ROHAN J

Reg No : 212223040171

import pandas as pd
df = pd.read_csv("Encoding_data.csv")
df.head()

![372034180-3c4708b7-1e8e-404f-bfa4-ad9828da4d8b](https://github.com/user-attachments/assets/8d2d2e08-7372-4c66-a155-1ea55dfe8d0d)

df.tail()

![372034344-d738d222-205f-4c42-b097-907ad1dd61a2](https://github.com/user-attachments/assets/c41c37b7-d640-440c-9f55-a24f93e822e9)

df.describe()

![372034567-b470ef07-1cd5-4484-b711-9f61520d7e56](https://github.com/user-attachments/assets/305a6fd4-42ef-435c-8a9f-3a53b574966a)

df.info()

![372034715-b0a38927-ebd6-4b90-abd7-5b21436f8901](https://github.com/user-attachments/assets/c4c30384-ffab-4d48-9a54-18756fd8d45f)

df.shape

![372034939-74d82d13-e1ef-4ed8-86e5-d4d855ab928f](https://github.com/user-attachments/assets/e451646d-de87-4219-b213-da9cfcce40f1)
df

![372035162-410e9f6d-3800-434a-84bd-efd07a06eb10](https://github.com/user-attachments/assets/1603b6c9-66eb-413e-a903-8280a7dcf716)
```
#ordinal encoder
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot', 'Warm','Cold']
oe=OrdinalEncoder(categories=[pm])
oe.fit_transform(df[["ord_2"]])
```
![372035305-30097aeb-f7f0-4098-b4da-0fd9f6fa2bd3](https://github.com/user-attachments/assets/c68b4200-a9a7-4e05-b760-275af3b7b4b7)
```
df['bo2']=oe.fit_transform(df[["ord_2"]])
df
```
![372035445-f4824aa3-72e2-4354-b39a-14d7bc6c5da8](https://github.com/user-attachments/assets/2efe1612-960e-41b0-a7ca-0196ac1e58f4)
```
#label Encoder
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![372035614-43d81ec8-49ab-43ef-888b-d7f097a39452](https://github.com/user-attachments/assets/9aa27b7e-fd40-4095-bf8d-bd45dcfcaf21)
```
#One hot encoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse = False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![372035923-b1ddad21-90b8-4159-9548-8eca78a7dae5](https://github.com/user-attachments/assets/d18c1ff7-e909-4760-96b1-5e171dbe0e1e)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![372036143-a0ea2434-a534-4afa-8b91-b17617c06e5a](https://github.com/user-attachments/assets/d59b2db1-eade-4972-8c74-f1d94517af7f)
```
pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
df= pd.read_csv("data.csv")
df
```
![372036425-421dc1e1-fcb5-4202-87da-4517a20956d6](https://github.com/user-attachments/assets/8e619a8b-216a-436c-b895-45e0003a4538)
```
#binary encoder
be = BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![372036654-9f44b784-9b99-41d2-8f29-39d3693cbbae](https://github.com/user-attachments/assets/f6191be0-aedd-460e-bea6-129bbd9af71c)
```
#target encoder
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new = te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![372036827-f91f18ce-7e69-4cee-81a3-6bcf54b48958](https://github.com/user-attachments/assets/74928b39-448d-4d6a-ae54-6296edd40c7f)
```
#Feature Transformation
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![372037136-3b1ec929-ad0e-47fb-b39a-03d9c228d6e1](https://github.com/user-attachments/assets/4e066113-16d9-4fa2-9844-e460e64c8a18)

df.info()

![372037288-327ed0d1-2f8f-41c5-9da9-125c79d5d1ff](https://github.com/user-attachments/assets/9bd54278-c6ba-438d-846d-12308289cb7a)

df.describe()

![372037451-5f5c8f7a-49be-48dc-86c1-400a0214618d](https://github.com/user-attachments/assets/4ca0d08b-743e-4df9-9afb-eb88f97e70e2)

df.size

![372037588-4af8996e-d807-4ba2-8def-1ddcddca8d4d](https://github.com/user-attachments/assets/528dffce-d22c-43ba-aa3e-c98f4c4f6aeb)

df.skew()
![372037729-8be208a6-db9f-4d23-abb3-7ccd8661225a](https://github.com/user-attachments/assets/3bc640f8-ade6-4ed5-8c90-9d651c81d12e)
```
np.log(df["Highly Positive Skew"])
```
![372037918-78aafe8a-29e8-4602-94cf-04d5a6412c76](https://github.com/user-attachments/assets/022971ef-d1c6-44b8-acf3-9d2697ccc6d1)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![372038134-11e16ca2-36cc-4bb4-9ae2-8602167b6df5](https://github.com/user-attachments/assets/eb8cb2dd-18fe-4b1c-9679-ed5e161bc71b)
```
np.sqrt(df["Highly Positive Skew"])
```
![372038342-2b3b9469-a49c-4285-8c53-6158025da034](https://github.com/user-attachments/assets/246a8c21-9308-4af5-aa60-f9b5df95b7e2)
```
np.square(df["Highly Positive Skew"])
```
![372038515-b2ee6d10-0ca8-4c43-8a5c-d1e425318246](https://github.com/user-attachments/assets/20c192f9-706c-4e36-9535-46ce7e691698)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![372038750-8715ef25-6571-4a90-b9a1-5d6ad36ccaf1](https://github.com/user-attachments/assets/7bd2b855-eeac-48ce-b358-165db224381f)
```
df["Moderate Negative Skew_yeojohnson"],parameters =stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![372039112-fda58aed-306d-4447-b66a-db7cdd0b233a](https://github.com/user-attachments/assets/a3677f74-f7f8-4b4c-9945-0225ce3e8a78)
```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![372039290-e4826368-b926-4d18-bc8f-7ff61372cc92](https://github.com/user-attachments/assets/58b483f8-beb9-4730-9ac2-bbafdb59cc5d)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])


sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![372039497-189e9be3-76c4-41ca-a7cf-63764cebec7b](https://github.com/user-attachments/assets/b95bf2fd-e5c4-4753-903e-fae1290ea114)

![372039731-9a74678c-3e7c-4d20-96cf-c95e1b37707a](https://github.com/user-attachments/assets/3d4d2509-114e-40f6-9c07-ba9bb3d0f6d9)
```
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![372039963-cdccbea2-1a33-4f3b-9ba3-95a12ac42751](https://github.com/user-attachments/assets/d63b9e4b-0a46-4fd3-a2e9-4452582810a4)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![372040112-c2a44890-9aee-4a59-808c-e5e43bd57817](https://github.com/user-attachments/assets/8b5db6a5-1dc2-4242-b76e-138ee6abd302)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![372040288-b42704ee-86d4-4dee-8d41-544235bf59e3](https://github.com/user-attachments/assets/4c3e829d-1bd1-4fd5-95fb-5199d6e583ab)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![372040456-13242ffc-05a2-467d-ba27-0c3555419b26](https://github.com/user-attachments/assets/918835ce-72be-4088-83cc-c1ad9d79b2e9)


      
# RESULT:
     
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully. 

       

