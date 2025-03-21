#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn')


# In[2]:


get_ipython().system('pip install matplotlib')


# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# In[4]:


#load the dataset
data = pd.read_csv("vehicle.csv")
print(data)


# In[5]:


data["cylinders"].value_counts()


# In[6]:


#identify categorical and numerical values
data.info()


# - Observation: Identifies data types (numerical/categorical) and detects missing values.
# 

# In[8]:


print(type(data))
print(data.shape)
print(data.size)


# In[9]:


len(data)


# In[10]:


#check missing values
data.isnull().sum()


# In[11]:


#check the duplicate values
duplicate_rows = data[data.duplicated()]
duplicate_rows


# In[12]:


#drop model and make columns
data = data.drop(columns=['make', 'model'])
data


# In[13]:


#visualazation
#Barplot for vehicle class
plt.figure(figsize=(20,5))
figure3 = sns.barplot(data = data, x = "vehicle_class",  y= "engine_size")
plt.xticks(rotation = 75)
plt.title("All Vehicle Class")
plt.xlabel("Vehicle Class")
plt.ylabel("Cars")
plt.bar_label(figure3.containers[0])
plt.show()


# In[14]:


#Barplot for Transmission
plt.figure(figsize=(20,5))
figure3 = sns.barplot(data = data, x = "transmission",  y= "engine_size")
plt.xticks(rotation = 75)
plt.title("All Vehicle Class")
plt.xlabel("Transmission")
plt.ylabel("Cars")
plt.bar_label(figure3.containers[0])
plt.show()


# In[15]:


#Boxplot and Histogram for engine size
fig, axes = plt.subplots(2, 1,figsize=(8,6), gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=data["engine_size"],ax=axes[0],color='yellow',width=0.5,orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Engine_Size")
sns.histplot(data["engine_size"],kde=True, ax=axes[1],color='blue',bins=30)
axes[1].set_title("Histogram with kde")
axes[1].set_xlabel("Engine_Size")
plt.tight_layout()
plt.show()


# In[16]:


#Boxplot and Histogram for fuel consumption city
fig, axes = plt.subplots(2, 1,figsize=(8,6), gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=data["fuel_consumption_city"],ax=axes[0],color='yellow',width=0.5,orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("fuel_consumption_city")
sns.histplot(data["fuel_consumption_city"],kde=True, ax=axes[1],color='blue',bins=30)
axes[1].set_title("Histogram with kde")
axes[1].set_xlabel("fuel_consumption_city")
plt.tight_layout()
plt.show()


# In[17]:


#one-hot-encoding for categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
data_encoded.to_csv("vehicle_encoded.csv", index=False)


# In[18]:


#Separate the continous columns and perform standard scaling
numerical_cols = data.select_dtypes(include=['number']).columns
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
data.to_csv("vehicle_scaled.csv", index=False)


# In[19]:


#concatenate the one-hot encoded categorical values with the scaled numerical columns
numerical_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(include=['object']).columns
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data[numerical_cols]), columns=numerical_cols)
data_encoded = pd.get_dummies(data[categorical_cols], drop_first=True)
data_final = pd.concat([data_scaled, data_encoded], axis=1)
data_final.to_csv("vehicle_final.csv", index=False)


# In[20]:


# Divide the data into x,y parts
X = data[['vehicle_class', 'engine_size', 'transmission', 'cylinders',
          'fuel_consumption_city', 'fuel_consumption_hwy',
          'fuel_consumption_comb (L/100 km)', 'fuel_type']]  

y = data['co2_emissions']
X = pd.get_dummies(X, drop_first=True)


# In[21]:


#To split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


#To fit and train a Linear Regression Model
model = LinearRegression()
model.fit(X_train,y_train)


# In[23]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')


# In[24]:


#To fit and train a Lasso Regression model
lasso_model = Lasso(alpha=0.1) 
lasso_model.fit(X_train, y_train)


# In[25]:


y_pred = lasso_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")


# In[26]:


#To fit and train a Ridge Regression model
ridge_model = Ridge(alpha=1.0) 
ridge_model.fit(X_train, y_train)


# In[27]:


y_pred = ridge_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")


# In[28]:


##To fit and train a Ridge Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[29]:


y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")


# In[30]:


#Deployment pkl file
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scale.pkl')


# In[31]:


joblib.dump(model, 'co2_emission_model.pkl')

