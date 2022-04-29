import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

passengers = pd.read_csv('passengers.csv')
#print(passengers)
# Update sex column to numerical
passengers['Sex'] = pd.Series(map(lambda a : 0 if a == 'male' else 1 , passengers.loc[:,'Sex']))
#print(passengers['Sex'])

#print(passengers['Age'])
passengers['Age'].fillna(value = passengers['Age'].mean(), inplace=True)

#print(passengers)
passengers['FirstClass'] = passengers['Pclass'].apply(lambda a: 1 if a == 1 else 0)

passengers['SecondClass'] = passengers['Pclass'].apply(lambda a: 1 if a == 2 else 0)
#print(passengers)

features = passengers[['Sex', 'Age', 'FirstClass','SecondClass']]
survival = passengers['Survived']

X_train, X_test, y_train, y_test = train_test_split(features, survival, test_size=0.33, random_state=42)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_train, y_train))

print(model.score(X_test, y_test))

print(list(zip(['Sex','Age','FirstClass','SecondClass'],model.coef_[0])))

Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Me = np.array([0.0,24.0,0.0,1.0])

sample_passengers = np.array([Jack, Rose, Me])

sample_passengers = scale.transform(sample_passengers)

print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))
