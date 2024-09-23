import pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
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
model_linear=LinearRegression()
model_ridge = Ridge()
model_nn = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000)
model_bagging = BaggingRegressor(base_estimator=model_linear, n_estimators=10)
estimators = [('linear', model_linear), ('ridge', model_ridge), ('mlp', model_nn)]
model_stacking = StackingRegressor(estimators=estimators, final_estimator=Ridge())
model_stacking.fit(X_train, y_train)
predictions_stacking = model_bagging.predict(X_test)
print(predictions_stacking)
print(r2_score(y_test, predictions_stacking))