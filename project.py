import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LogisticRegression    
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df=pd.read_excel("tel_cus_churn.xlsx")
print(df)
t=df.isna().sum()
print(t)
t=df.iloc[:21]
print(t)
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
print(df)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
print(df)
df['TotalCharges']=df['TotalCharges'].fillna(df['TotalCharges'].mean())
print(df)
df.drop("customerID",axis=1,inplace=True)
print(df)
sns.countplot(x="Churn", data=df)
plt.show()
sns.countplot(x="Contract", hue="Churn", data=df)
plt.show()
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.show()

model=LabelEncoder()
binary_col=['gender','Partner','Dependents','PhoneService','PaperlessBilling','Churn']
for col in binary_col:
    df[col]=model.fit_transform(df[col])
print(df)

x=df[[
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]]
y=df['Churn']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(y_pred)
print(accuracy_score(y_test,y_pred))
import joblib
joblib.dump(model,"churn_model.pkl")
print("model saved successfully")
      
