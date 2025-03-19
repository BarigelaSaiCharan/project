#!/usr/bin/env python
# coding: utf-8

# In[108]:


get_ipython().system('pip install scikit-learn')


# In[109]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[110]:


data = pd.read_csv("vehicle.csv")
print(data)


# In[111]:


data["cylinders"].value_counts()


# In[112]:


data.info()


# - Observation: Identifies data types (numerical/categorical) and detects missing values.
# 

# In[119]:


print(type(data))
print(data.shape)
print(data.size)


# In[120]:


len(data)


# In[121]:


data.describe()


# In[122]:


data.reset_index(drop=True, inplace=True)
data.head()


# In[123]:


data.isnull().sum()


# - Observation: Identifies which columns require data cleaning

# In[125]:


X = data[['engine_size', 'cylinders', 'fuel_consumption_city', 'fuel_type']]  
y = data['co2_emissions']
X = pd.get_dummies(X, drop_first=True)  


# In[126]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[127]:


model = LinearRegression()
model.fit(X_train,y_train)


# In[130]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')


# In[131]:


median_engine_size = data["engine_size"].median()
mean_engine_size = data["engine_size"].mean()
print("Median of engine_size: ",median_engine_size)
print("Mean of engine_size: ",mean_engine_size)


# In[132]:


correlation_matrix = data.select_dtypes(include=['number']).corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="cool", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of CO₂ Emissions and Other Features")
plt.show()


# #### Observation
# - engine_size, fuel_consumption, or vehicle_weight might show strong positive correlations with CO₂ emissions.
# - Negative correlations might indicate inverse relationships (e.g., fuel efficiency vs. emissions).

# In[134]:


plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel('Actual CO2 Emissions')
plt.ylabel('Predicted CO2 Emissions')
plt.title('Actual vs Predicted CO2 Emissions')
plt.show()


# In[135]:


data['engine_size'] = data['engine_size'].fillna(median_engine_size)
data.isnull().sum()


# In[136]:


mean_engine_size = data["engine_size"].mean()
print("Mean of engine_size: ",mean_engine_size)


# In[137]:


print(data["model"].value_counts())
mode_model = data["model"].mode()[0]
print(mode_model)


# In[138]:


import joblib
joblib.dump(model, 'co2_emission_model.pkl')


# In[139]:


from sklearn.preprocessing import StandardScaler
import joblib
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')


# In[140]:


plt.figure(figsize=(20,5))
figure3 = sns.barplot(data = data, x = "vehicle_class",  y= "engine_size")
plt.xticks(rotation = 75)
plt.title("All Vehicle Class")
plt.xlabel("Vehicle Class")
plt.ylabel("Cars")
plt.bar_label(figure3.containers[0])
plt.show()


# In[141]:


fig, axes = plt.subplots(2, 1,figsize=(8,6), gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=data["engine_size"],ax=axes[0],color='red',width=0.5,orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Engine Size")
sns.histplot(data["engine_size"],kde=True, ax=axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with kde")
axes[1].set_xlabel("Engine Size")
plt.tight_layout()
plt.show()


# In[142]:


fig, axes = plt.subplots(2, 1,figsize=(8,6), gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=data["model"],ax=axes[0],color='yellow',width=0.5,orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Model")
sns.histplot(data["model"],kde=True, ax=axes[1],color='blue',bins=30)
axes[1].set_title("Histogram with kde")
axes[1].set_xlabel("Model")
plt.tight_layout()
plt.show()


# In[143]:


plt.figure(figsize=(5,5))
plt.boxplot(data["engine_size"],vert= False)


# In[144]:


plt.figure(figsize=(6,3))
boxplot_data = plt.boxplot(data["engine_size"], vert=False)
[x.get_xdata() for x in boxplot_data['boxes']]


# #### Observation
# - Extracts data points from the boxplot to analyze outliers

# In[146]:


data["engine_size"].describe()


# In[147]:


import scipy.stats as stats
plt.figure(figsize=(8, 6))
stats.probplot(data["engine_size"],dist="norm",plot=plt)
plt.title("Q-Q Plot for Outliers Detection",fontsize=14)
plt.xlabel("Theretical Quantiles",fontsize=12)


# In[148]:


sns.violinplot(data=data["engine_size"],color='lightgreen')
plt.title("Violin Plot")
plt.show()

