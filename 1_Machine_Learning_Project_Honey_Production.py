import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")
print(df.head())

prod_per_year = df.groupby('year').totalprod.mean().reset_index()
print(prod_per_year)

X = prod_per_year['year']
#print(X)
X = X.values.reshape(-1, 1)
print(type(X))

y = prod_per_year['totalprod']
plt.scatter(X,y)

regr = linear_model.LinearRegression()
regr.fit(X,y)

print(regr.coef_[0]) #it gives me m parameter of y formula and we have [0] cause regr.coef_ gives list
print(regr.intercept_) #it gives me b parameter of y formula

y_predict = regr.predict(X)
plt.plot(X,y_predict)
plt.show()

X_future = np.array(range(2013,2051))
X_future = X_future.reshape(-1, 1) #reshape for scikit-learn

future_predict = regr.predict(X_future)
plt.plot(X_future, future_predict)
plt.show()
