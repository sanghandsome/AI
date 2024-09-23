import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import linear_model
import csv

data = pd.read_csv('housing.csv')
data.fillna({'total_bedrooms': data['total_bedrooms'].mean()}, inplace=True)
mapping = {'<1H OCEAN': 1,'INLAND': 2,'NEAR OCEAN': 3,'NEAR BAY':4,'ISLAND':5}
data['ocean_proximity'] = data['ocean_proximity'].map(mapping)
X=data[["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","ocean_proximity"]]
y=data[["median_house_value"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_ridge = Ridge()
model_ridge.fit(X_train,y_train)
predictions_ridge = model_ridge.predict(X_test)
print(predictions_ridge)
print(r2_score(y_test, predictions_ridge))